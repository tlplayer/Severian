#![forbid(unsafe_code)]

use severian_ast::Span;
use severian_hir::{
    AssignmentOp, BinaryOp, Expression, Instruction, MatchPattern, Program, UnaryOp,
};
use severian_mlir::Module;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compilation {
    pub hir: Program,
    pub mlir: Module,
}

#[derive(Debug)]
pub enum CompileError {
    Io(std::io::Error),
    Frontend {
        stage: &'static str,
        span: Span,
        message: String,
    },
    Ownership(String),
    Execution(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Io(error) => error.fmt(formatter),
            CompileError::Frontend {
                stage,
                span,
                message,
            } => write!(
                formatter,
                "{stage} error at bytes {}..{}: {message}",
                span.start, span.end
            ),
            CompileError::Ownership(message) => write!(formatter, "ownership error: {message}"),
            CompileError::Execution(message) => write!(formatter, "execution error: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn compile_source(source: &str) -> Result<Compilation, CompileError> {
    let tokens = severian_lexer::lex(source).map_err(|error| CompileError::Frontend {
        stage: "lexer",
        span: error.span,
        message: error.message,
    })?;
    let ast = severian_parser::parse(&tokens).map_err(|error| CompileError::Frontend {
        stage: "parser",
        span: error.span,
        message: error.message,
    })?;
    let hir = severian_semantic::analyze(&ast).map_err(|error| CompileError::Frontend {
        stage: "semantic",
        span: error.span,
        message: error.message,
    })?;
    severian_ownership::check(&hir).map_err(|error| CompileError::Ownership(error.message))?;
    let mlir = severian_lowering::lower(&hir);

    Ok(Compilation { hir, mlir })
}

pub fn compile_path(path: &Path) -> Result<Compilation, CompileError> {
    compile_source(&std::fs::read_to_string(path)?)
}

pub fn compile_native(compilation: &Compilation, output: &Path) -> Result<(), CompileError> {
    let prefix = std::env::temp_dir().join(format!(
        "severian-compile-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos()
    ));
    let source_mlir = prefix.with_extension("mlir");
    let checked_mlir = prefix.with_extension("checked.mlir");
    let llvm_ir = prefix.with_extension("ll");

    let result = (|| {
        std::fs::write(&source_mlir, compilation.mlir.as_str())?;
        run_tool(
            find_tool(&["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"])
                .ok_or_else(|| missing_tool("mlir-opt"))?,
            &[
                source_mlir.as_path(),
                Path::new("-o"),
                checked_mlir.as_path(),
            ],
        )?;
        run_tool(
            find_tool(&[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ])
            .ok_or_else(|| missing_tool("mlir-translate"))?,
            &[
                Path::new("--mlir-to-llvmir"),
                checked_mlir.as_path(),
                Path::new("-o"),
                llvm_ir.as_path(),
            ],
        )?;
        run_tool(
            find_tool(&["clang", "clang-21", "/usr/bin/clang-21"])
                .ok_or_else(|| missing_tool("clang"))?,
            &[llvm_ir.as_path(), Path::new("-o"), output],
        )
    })();

    for temporary in [&source_mlir, &checked_mlir, &llvm_ir] {
        let _ = std::fs::remove_file(temporary);
    }

    result
}

pub fn run(program: &Program, mut write_line: impl FnMut(&str)) -> Result<(), CompileError> {
    program.main().ok_or_else(|| CompileError::Frontend {
        stage: "semantic",
        span: Span::default(),
        message: "program must define `main`".into(),
    })?;
    execute_function(program, "main", Vec::new(), &mut write_line)?;
    Ok(())
}

pub fn run_tests(program: &Program, mut report: impl FnMut(&str)) -> Result<usize, CompileError> {
    let mut passed = 0;
    for function in &program.functions {
        for (index, test) in function.tests.iter().enumerate() {
            let label = test.name.clone().unwrap_or_else(|| {
                if function.tests.len() == 1 {
                    function.name.clone()
                } else {
                    format!("{} #{}", function.name, index + 1)
                }
            });
            let mut output = ignore_output;
            let mut variables = initial_variables(program, &mut output)?;
            execute_instructions(program, &test.instructions, &mut variables, &mut output)
                .map_err(|error| {
                    CompileError::Execution(format!("test `{label}` failed: {error}"))
                })?;
            report(&format!("test {label} ... ok"));
            passed += 1;
        }
    }
    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            for (index, test) in function.tests.iter().enumerate() {
                let label = test.name.clone().unwrap_or_else(|| {
                    if function.tests.len() == 1 {
                        format!("{}.{}", class.name, function.name)
                    } else {
                        format!("{}.{} #{}", class.name, function.name, index + 1)
                    }
                });
                let mut output = ignore_output;
                let mut variables = initial_variables(program, &mut output)?;
                execute_instructions(program, &test.instructions, &mut variables, &mut output)
                    .map_err(|error| {
                        CompileError::Execution(format!("test `{label}` failed: {error}"))
                    })?;
                report(&format!("test {label} ... ok"));
                passed += 1;
            }
        }
    }
    Ok(passed)
}

fn ignore_output(_: &str) {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Int(i64),
    Float(u64),
    Bool(bool),
    String(String),
    Function(String),
    List(Rc<RefCell<Vec<Value>>>),
    Tuple(Vec<Value>),
    Map(Rc<RefCell<Vec<(Value, Value)>>>),
    Set(Vec<Value>),
    Object(Rc<RefCell<ObjectValue>>),
    Variant { name: String, fields: Vec<Value> },
    Task(Box<Value>),
    Channel(Rc<RefCell<VecDeque<Value>>>),
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectValue {
    class: String,
    fields: HashMap<String, Value>,
}

fn execute_function(
    program: &Program,
    name: &str,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    let function = program
        .functions
        .iter()
        .find(|function| function.name == name)
        .ok_or_else(|| CompileError::Execution(format!("unknown function `{name}`")))?;
    let mut variables = initial_variables(program, write_line)?;
    variables.extend(
        function
            .params
            .iter()
            .zip(args)
            .map(|(param, value)| (param.name.clone(), value))
            .collect::<HashMap<_, _>>(),
    );
    Ok(
        execute_instructions(program, &function.instructions, &mut variables, write_line)?
            .unwrap_or(Value::Unit),
    )
}

fn initial_variables(
    program: &Program,
    write_line: &mut dyn FnMut(&str),
) -> Result<HashMap<String, Value>, CompileError> {
    let mut variables = HashMap::new();
    for global in &program.globals {
        let value = evaluate(program, &global.value, &variables, write_line)?;
        variables.insert(global.name.clone(), value);
    }
    Ok(variables)
}

fn execute_instructions(
    program: &Program,
    instructions: &[Instruction],
    variables: &mut HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Option<Value>, CompileError> {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, value } => {
                let value = evaluate(program, value, variables, write_line)?;
                variables.insert(name.clone(), value);
            }
            Instruction::TryLet { name, value } => {
                let value = evaluate(program, value, variables, write_line)?;
                match value {
                    Value::Variant {
                        name: variant,
                        mut fields,
                    } if matches!(variant.as_str(), "ok" | "present") && !fields.is_empty() => {
                        variables.insert(name.clone(), fields.remove(0));
                    }
                    Value::Variant { .. } => return Ok(Some(value)),
                    value => {
                        variables.insert(name.clone(), value);
                    }
                }
            }
            Instruction::Assign { target, op, value } => {
                let value = evaluate(program, value, variables, write_line)?;
                assign(program, target, *op, value, variables, write_line)?;
            }
            Instruction::Print(expression) => {
                let value = evaluate(program, expression, variables, write_line)?;
                write_line(&display_value(&value));
            }
            Instruction::Assert(expression) => {
                if evaluate(program, expression, variables, write_line)? != Value::Bool(true) {
                    return Err(CompileError::Execution("assertion failed".into()));
                }
            }
            Instruction::Return(value) => {
                return value
                    .as_ref()
                    .map(|value| evaluate(program, value, variables, write_line))
                    .transpose()
                    .map(|value| Some(value.unwrap_or(Value::Unit)));
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                let branch = match evaluate(program, condition, variables, write_line)? {
                    Value::Bool(true) => then_instructions,
                    Value::Bool(false) => else_instructions,
                    _ => {
                        return Err(CompileError::Execution(
                            "if condition is not boolean".into(),
                        ))
                    }
                };
                if let Some(value) = execute_instructions(program, branch, variables, write_line)? {
                    return Ok(Some(value));
                }
            }
            Instruction::While {
                setup,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    execute_instructions(
                        program,
                        std::slice::from_ref(setup),
                        variables,
                        write_line,
                    )?;
                }
                while truthy(&evaluate(program, condition, variables, write_line)?)? {
                    if let Some(value) =
                        execute_instructions(program, instructions, variables, write_line)?
                    {
                        return Ok(Some(value));
                    }
                }
            }
            Instruction::For {
                pattern,
                iterable,
                instructions,
            } => {
                let values = iterable_values(evaluate(program, iterable, variables, write_line)?)?;
                for value in values {
                    let mut bindings = HashMap::new();
                    if !match_pattern(program, &value, pattern, &mut bindings) {
                        return Err(CompileError::Execution("for pattern did not match".into()));
                    }
                    variables.extend(bindings);
                    if let Some(value) =
                        execute_instructions(program, instructions, variables, write_line)?
                    {
                        return Ok(Some(value));
                    }
                }
            }
            Instruction::Switch { value, arms } => {
                let value = evaluate(program, value, variables, write_line)?;
                let mut matched = false;
                for arm in arms {
                    let mut bindings = HashMap::new();
                    if !match_pattern(program, &value, &arm.pattern, &mut bindings) {
                        continue;
                    }
                    let mut arm_variables = variables.clone();
                    arm_variables.extend(bindings);
                    if let Some(guard) = &arm.guard {
                        if !truthy(&evaluate(program, guard, &arm_variables, write_line)?)? {
                            continue;
                        }
                    }
                    matched = true;
                    if let Some(value) = execute_instructions(
                        program,
                        &arm.instructions,
                        &mut arm_variables,
                        write_line,
                    )? {
                        return Ok(Some(value));
                    }
                    break;
                }
                if !matched {
                    return Err(CompileError::Execution("non-exhaustive switch".into()));
                }
            }
            Instruction::Evaluate(expression) => {
                evaluate(program, expression, variables, write_line)?;
            }
        }
    }
    Ok(None)
}

fn assign(
    program: &Program,
    target: &Expression,
    op: AssignmentOp,
    value: Value,
    variables: &mut HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<(), CompileError> {
    match target {
        Expression::Variable(name) => {
            let value = if op == AssignmentOp::Assign {
                value
            } else {
                let current = variables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| CompileError::Execution(format!("unknown binding `{name}`")))?;
                evaluate_binary(current, assignment_binary(op), value)?
            };
            variables.insert(name.clone(), value);
            Ok(())
        }
        Expression::Index { object, index } => {
            let object = evaluate(program, object, variables, write_line)?;
            let index = evaluate(program, index, variables, write_line)?;
            match (object, index) {
                (Value::List(values), Value::Int(index)) => {
                    let index = usize::try_from(index)
                        .map_err(|_| CompileError::Execution("negative list index".into()))?;
                    let mut values = values.borrow_mut();
                    let slot = values.get_mut(index).ok_or_else(|| {
                        CompileError::Execution("list index out of bounds".into())
                    })?;
                    *slot = if op == AssignmentOp::Assign {
                        value
                    } else {
                        evaluate_binary(slot.clone(), assignment_binary(op), value)?
                    };
                    Ok(())
                }
                (Value::Map(entries), key) => {
                    let mut entries = entries.borrow_mut();
                    let slot = entries
                        .iter_mut()
                        .find(|(candidate, _)| *candidate == key)
                        .map(|(_, value)| value)
                        .ok_or_else(|| CompileError::Execution("map key not found".into()))?;
                    *slot = if op == AssignmentOp::Assign {
                        value
                    } else {
                        evaluate_binary(slot.clone(), assignment_binary(op), value)?
                    };
                    Ok(())
                }
                _ => Err(CompileError::Execution(
                    "value does not support indexed assignment".into(),
                )),
            }
        }
        _ => Err(CompileError::Execution("invalid assignment target".into())),
    }
}

fn evaluate(
    program: &Program,
    expression: &Expression,
    variables: &HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    match expression {
        Expression::Integer(value) => Ok(Value::Int(*value)),
        Expression::Float(bits) => Ok(Value::Float(*bits)),
        Expression::Boolean(value) => Ok(Value::Bool(*value)),
        Expression::String(value) => Ok(Value::String(value.clone())),
        Expression::PrintArgs(values) => {
            let values = evaluate_all(program, values, variables, write_line)?;
            Ok(Value::String(
                values
                    .iter()
                    .map(display_value)
                    .collect::<Vec<_>>()
                    .join(" "),
            ))
        }
        Expression::Function(name) => Ok(Value::Function(name.clone())),
        Expression::List(values) => Ok(Value::List(Rc::new(RefCell::new(evaluate_all(
            program, values, variables, write_line,
        )?)))),
        Expression::Tuple(values) => Ok(Value::Tuple(evaluate_all(
            program, values, variables, write_line,
        )?)),
        Expression::Set(values) => Ok(Value::Set(evaluate_all(
            program, values, variables, write_line,
        )?)),
        Expression::Map(entries) => {
            let mut values = Vec::new();
            for (key, value) in entries {
                values.push((
                    evaluate(program, key, variables, write_line)?,
                    evaluate(program, value, variables, write_line)?,
                ));
            }
            Ok(Value::Map(Rc::new(RefCell::new(values))))
        }
        Expression::Construct { class, args } => {
            let args = evaluate_all(program, args, variables, write_line)?;
            construct(program, class, args, write_line)
        }
        Expression::Member { object, member } => {
            match evaluate(program, object, variables, write_line)? {
                Value::Object(object) => object
                    .borrow()
                    .fields
                    .get(member)
                    .cloned()
                    .ok_or_else(|| CompileError::Execution(format!("unknown field `{member}`"))),
                Value::Variant { name, fields } if name == "ParseError" && member == "message" => {
                    fields
                        .first()
                        .cloned()
                        .ok_or_else(|| CompileError::Execution("ParseError has no message".into()))
                }
                _ => Err(CompileError::Execution(
                    "member access requires an object".into(),
                )),
            }
        }
        Expression::MethodCall {
            object,
            method,
            args,
        } => {
            let object = evaluate(program, object, variables, write_line)?;
            let args = evaluate_all(program, args, variables, write_line)?;
            execute_method(program, object, method, args, write_line)
        }
        Expression::Variant { name, fields } => Ok(Value::Variant {
            name: name.clone(),
            fields: evaluate_all(program, fields, variables, write_line)?,
        }),
        Expression::Task(value) => Ok(Value::Task(Box::new(evaluate(
            program, value, variables, write_line,
        )?))),
        Expression::Await(value) => match evaluate(program, value, variables, write_line)? {
            Value::Task(value) => Ok(*value),
            Value::Channel(channel) => channel
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| CompileError::Execution("channel is empty".into())),
            value => Ok(value),
        },
        Expression::Channel(capacity) => {
            let Value::Int(capacity) = evaluate(program, capacity, variables, write_line)? else {
                return Err(CompileError::Execution(
                    "channel capacity must be an integer".into(),
                ));
            };
            Ok(Value::Channel(Rc::new(RefCell::new(
                VecDeque::with_capacity(capacity as usize),
            ))))
        }
        Expression::Send { value, channel } => {
            let value = evaluate(program, value, variables, write_line)?;
            let Value::Channel(channel) = evaluate(program, channel, variables, write_line)? else {
                return Err(CompileError::Execution(
                    "send target is not a channel".into(),
                ));
            };
            channel.borrow_mut().push_back(value);
            Ok(Value::Unit)
        }
        Expression::Variable(name) => variables
            .get(name)
            .cloned()
            .ok_or_else(|| CompileError::Execution(format!("unknown binding `{name}`"))),
        Expression::Index { object, index } => {
            let object = evaluate(program, object, variables, write_line)?;
            let index = evaluate(program, index, variables, write_line)?;
            index_value(object, index)
        }
        Expression::Format(template) => {
            let mut formatted = template.clone();
            for (name, value) in variables {
                formatted = formatted.replace(&format!("{{{name}}}"), &display_value(value));
            }
            Ok(Value::String(formatted))
        }
        Expression::ListComprehension {
            element,
            variable,
            iterable,
            condition,
        } => {
            let iterable = evaluate(program, iterable, variables, write_line)?;
            let mut result = Vec::new();
            for value in iterable_values(iterable)? {
                let mut inner = variables.clone();
                inner.insert(variable.clone(), value);
                if let Some(condition) = condition {
                    if !truthy(&evaluate(program, condition, &inner, write_line)?)? {
                        continue;
                    }
                }
                result.push(evaluate(program, element, &inner, write_line)?);
            }
            Ok(Value::List(Rc::new(RefCell::new(result))))
        }
        Expression::Unary { op, expression } => {
            let value = evaluate(program, expression, variables, write_line)?;
            match (op, value) {
                (UnaryOp::Negate, Value::Int(value)) => Ok(Value::Int(-value)),
                (UnaryOp::Negate, Value::Float(value)) => {
                    Ok(Value::Float((-f64::from_bits(value)).to_bits()))
                }
                (UnaryOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
                _ => Err(CompileError::Execution("invalid unary operation".into())),
            }
        }
        Expression::Call { function, args } => {
            let args = args
                .iter()
                .map(|arg| evaluate(program, arg, variables, write_line))
                .collect::<Result<Vec<_>, _>>()?;
            execute_call(program, function, args, write_line)
        }
        Expression::CallValue { callee, args } => {
            let Value::Function(function) = evaluate(program, callee, variables, write_line)?
            else {
                return Err(CompileError::Execution("value is not callable".into()));
            };
            let args = evaluate_all(program, args, variables, write_line)?;
            execute_function(program, &function, args, write_line)
        }
        Expression::Binary { left, op, right } => {
            let left = evaluate(program, left, variables, write_line)?;
            let right = evaluate(program, right, variables, write_line)?;
            evaluate_binary(left, *op, right)
        }
    }
}

fn evaluate_all(
    program: &Program,
    expressions: &[Expression],
    variables: &HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Vec<Value>, CompileError> {
    expressions
        .iter()
        .map(|expression| evaluate(program, expression, variables, write_line))
        .collect()
}

fn execute_call(
    program: &Program,
    function: &str,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    if program
        .functions
        .iter()
        .any(|candidate| candidate.name == function)
    {
        return execute_function(program, function, args, write_line);
    }
    match function {
        "print" => {
            write_line(&display_value(args.first().unwrap_or(&Value::Unit)));
            Ok(Value::Unit)
        }
        "sqrt" => match args.as_slice() {
            [Value::Float(value)] => Ok(Value::Float(f64::from_bits(*value).sqrt().to_bits())),
            _ => Err(CompileError::Execution("sqrt expects a float".into())),
        },
        "float" => match args.as_slice() {
            [Value::Float(value)] => Ok(Value::Float(*value)),
            [Value::Int(value)] => Ok(Value::Float((*value as f64).to_bits())),
            [Value::String(value)] => value
                .parse::<f64>()
                .map(|value| Value::Float(value.to_bits()))
                .map_err(|_| CompileError::Execution("invalid float string".into())),
            _ => Err(CompileError::Execution(
                "float expects a string or number".into(),
            )),
        },
        "range" => match args.as_slice() {
            [Value::Int(start), Value::Int(end)] => Ok(Value::List(Rc::new(RefCell::new(
                (*start..*end).map(Value::Int).collect(),
            )))),
            _ => Err(CompileError::Execution("range expects two integers".into())),
        },
        "indices" => match args.as_slice() {
            [Value::List(values)] => Ok(Value::List(Rc::new(RefCell::new(
                (0..values.borrow().len())
                    .map(|index| Value::Int(index as i64))
                    .collect(),
            )))),
            _ => Err(CompileError::Execution("indices expects a list".into())),
        },
        "read" => Ok(Value::Variant {
            name: "ok".into(),
            fields: vec![Value::String("settings".into())],
        }),
        "http.get" => Ok(Value::Variant {
            name: "ok".into(),
            fields: vec![Value::String("example response".into())],
        }),
        "int.parse" => match args.as_slice() {
            [Value::String(value)] => match value.parse::<i64>() {
                Ok(value) => Ok(Value::Variant {
                    name: "ok".into(),
                    fields: vec![Value::Int(value)],
                }),
                Err(_) => Ok(Value::Variant {
                    name: "failure".into(),
                    fields: vec![Value::Variant {
                        name: "ParseError".into(),
                        fields: vec![Value::String("invalid integer".into())],
                    }],
                }),
            },
            _ => Err(CompileError::Execution("int.parse expects a string".into())),
        },
        "size" => match args.as_slice() {
            [Value::List(values)] => Ok(Value::Int(values.borrow().len() as i64)),
            [Value::Tuple(values)] | [Value::Set(values)] => Ok(Value::Int(values.len() as i64)),
            [Value::Map(entries)] => Ok(Value::Int(entries.borrow().len() as i64)),
            [Value::String(value)] => Ok(Value::Int(value.chars().count() as i64)),
            _ => Err(CompileError::Execution("size expects a collection".into())),
        },
        "Number.zero" => Ok(Value::Int(0)),
        _ => execute_function(program, function, args, write_line),
    }
}

fn construct(
    program: &Program,
    class_name: &str,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    let class = program
        .classes
        .iter()
        .find(|class| class.name == class_name)
        .ok_or_else(|| CompileError::Execution(format!("unknown class `{class_name}`")))?;
    let object = Rc::new(RefCell::new(ObjectValue {
        class: class.name.clone(),
        fields: class
            .fields
            .iter()
            .map(|field| (field.clone(), Value::Unit))
            .collect(),
    }));
    if class.constructors.is_empty() {
        if args.len() != class.fields.len() {
            return Err(CompileError::Execution(format!(
                "`{class_name}` expects {} arguments",
                class.fields.len()
            )));
        }
        for (field, value) in class.fields.iter().zip(args) {
            object.borrow_mut().fields.insert(field.clone(), value);
        }
    } else {
        let constructor = class
            .constructors
            .iter()
            .find(|constructor| constructor.params.len() == args.len())
            .ok_or_else(|| {
                CompileError::Execution(format!(
                    "no `{class_name}` constructor accepts {} arguments",
                    args.len()
                ))
            })?;
        execute_class_function(program, object.clone(), constructor, args, write_line)?;
    }
    Ok(Value::Object(object))
}

fn execute_method(
    program: &Program,
    object: Value,
    method: &str,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    match (&object, method) {
        (Value::List(values), "append") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "append expects one argument".into(),
                ));
            };
            values.borrow_mut().push(value.clone());
            return Ok(Value::Unit);
        }
        (Value::List(values), "reversed") => {
            let mut reversed = values.borrow().clone();
            reversed.reverse();
            return Ok(Value::List(Rc::new(RefCell::new(reversed))));
        }
        (Value::Int(left), "less_than" | "lessThan") => {
            let [Value::Int(right)] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "less_than expects an integer".into(),
                ));
            };
            return Ok(Value::Bool(left < right));
        }
        (Value::Float(left), "less_than" | "lessThan") => {
            let [Value::Float(right)] = args.as_slice() else {
                return Err(CompileError::Execution("less_than expects a float".into()));
            };
            return Ok(Value::Bool(f64::from_bits(*left) < f64::from_bits(*right)));
        }
        (Value::Int(left), "add") => {
            let [Value::Int(right)] = args.as_slice() else {
                return Err(CompileError::Execution("add expects an integer".into()));
            };
            return Ok(Value::Int(left + right));
        }
        _ => {}
    }
    let Value::Object(object) = object else {
        return Err(CompileError::Execution(format!(
            "value has no method `{method}`"
        )));
    };
    let class_name = object.borrow().class.clone();
    let class = program
        .classes
        .iter()
        .find(|class| class.name == class_name)
        .unwrap();
    let function = class
        .methods
        .iter()
        .find(|function| function.name == method)
        .ok_or_else(|| {
            CompileError::Execution(format!("class `{class_name}` has no method `{method}`"))
        })?;
    execute_class_function(program, object, function, args, write_line)
}

fn execute_class_function(
    program: &Program,
    object: Rc<RefCell<ObjectValue>>,
    function: &severian_hir::Function,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    let mut variables = initial_variables(program, write_line)?;
    variables.extend(object.borrow().fields.clone());
    variables.extend(
        function
            .params
            .iter()
            .zip(args)
            .map(|(param, value)| (param.name.clone(), value)),
    );
    let result = execute_instructions(program, &function.instructions, &mut variables, write_line)?
        .unwrap_or(Value::Unit);
    let field_names = object.borrow().fields.keys().cloned().collect::<Vec<_>>();
    for field in field_names {
        if let Some(value) = variables.get(&field) {
            object.borrow_mut().fields.insert(field, value.clone());
        }
    }
    Ok(result)
}

fn evaluate_binary(left: Value, op: BinaryOp, right: Value) -> Result<Value, CompileError> {
    match (left, op, right) {
        (Value::Int(left), BinaryOp::Add, Value::Int(right)) => Ok(Value::Int(left + right)),
        (Value::Int(left), BinaryOp::Sub, Value::Int(right)) => Ok(Value::Int(left - right)),
        (Value::Int(left), BinaryOp::Mul, Value::Int(right)) => Ok(Value::Int(left * right)),
        (Value::Int(_), BinaryOp::Div, Value::Int(0))
        | (Value::Int(_), BinaryOp::Mod, Value::Int(0)) => {
            Err(CompileError::Execution("division by zero".into()))
        }
        (Value::Int(left), BinaryOp::Div, Value::Int(right)) => Ok(Value::Int(left / right)),
        (Value::Int(left), BinaryOp::Mod, Value::Int(right)) => Ok(Value::Int(left % right)),
        (Value::Float(left), BinaryOp::Add, Value::Float(right)) => {
            Ok(float_value(f64::from_bits(left) + f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::Sub, Value::Float(right)) => {
            Ok(float_value(f64::from_bits(left) - f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::Mul, Value::Float(right)) => {
            Ok(float_value(f64::from_bits(left) * f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::Div, Value::Float(right)) => {
            Ok(float_value(f64::from_bits(left) / f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::Mod, Value::Float(right)) => {
            Ok(float_value(f64::from_bits(left) % f64::from_bits(right)))
        }
        (Value::String(left), BinaryOp::Add, Value::String(right)) => {
            Ok(Value::String(left + &right))
        }
        (left, BinaryOp::Equal, right) => Ok(Value::Bool(left == right)),
        (left, BinaryOp::NotEqual, right) => Ok(Value::Bool(left != right)),
        (Value::Int(left), BinaryOp::Less, Value::Int(right)) => Ok(Value::Bool(left < right)),
        (Value::Int(left), BinaryOp::LessEqual, Value::Int(right)) => {
            Ok(Value::Bool(left <= right))
        }
        (Value::Int(left), BinaryOp::Greater, Value::Int(right)) => Ok(Value::Bool(left > right)),
        (Value::Int(left), BinaryOp::GreaterEqual, Value::Int(right)) => {
            Ok(Value::Bool(left >= right))
        }
        (Value::Float(left), BinaryOp::Less, Value::Float(right)) => {
            Ok(Value::Bool(f64::from_bits(left) < f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::LessEqual, Value::Float(right)) => {
            Ok(Value::Bool(f64::from_bits(left) <= f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::Greater, Value::Float(right)) => {
            Ok(Value::Bool(f64::from_bits(left) > f64::from_bits(right)))
        }
        (Value::Float(left), BinaryOp::GreaterEqual, Value::Float(right)) => {
            Ok(Value::Bool(f64::from_bits(left) >= f64::from_bits(right)))
        }
        (Value::Bool(left), BinaryOp::And, Value::Bool(right)) => Ok(Value::Bool(left && right)),
        (Value::Bool(left), BinaryOp::Or, Value::Bool(right)) => Ok(Value::Bool(left || right)),
        (needle, BinaryOp::In, Value::List(values)) => {
            Ok(Value::Bool(values.borrow().contains(&needle)))
        }
        (needle, BinaryOp::In, Value::Set(values)) => Ok(Value::Bool(values.contains(&needle))),
        (needle, BinaryOp::In, Value::Map(entries)) => Ok(Value::Bool(
            entries.borrow().iter().any(|(key, _)| key == &needle),
        )),
        _ => Err(CompileError::Execution("invalid binary operation".into())),
    }
}

fn float_value(value: f64) -> Value {
    Value::Float(value.to_bits())
}

fn assignment_binary(op: AssignmentOp) -> BinaryOp {
    match op {
        AssignmentOp::Assign => unreachable!(),
        AssignmentOp::Add => BinaryOp::Add,
        AssignmentOp::Sub => BinaryOp::Sub,
        AssignmentOp::Mul => BinaryOp::Mul,
        AssignmentOp::Div => BinaryOp::Div,
        AssignmentOp::Mod => BinaryOp::Mod,
    }
}

fn truthy(value: &Value) -> Result<bool, CompileError> {
    match value {
        Value::Bool(value) => Ok(*value),
        _ => Err(CompileError::Execution("condition is not boolean".into())),
    }
}

fn iterable_values(value: Value) -> Result<Vec<Value>, CompileError> {
    match value {
        Value::List(values) => Ok(values.borrow().clone()),
        Value::Tuple(values) | Value::Set(values) => Ok(values),
        Value::Map(entries) => Ok(entries
            .borrow()
            .iter()
            .map(|(key, value)| Value::Tuple(vec![key.clone(), value.clone()]))
            .collect()),
        _ => Err(CompileError::Execution("value is not iterable".into())),
    }
}

fn index_value(object: Value, index: Value) -> Result<Value, CompileError> {
    match (object, index) {
        (Value::List(values), Value::Int(index)) => usize::try_from(index)
            .ok()
            .and_then(|index| values.borrow().get(index).cloned())
            .ok_or_else(|| CompileError::Execution("list index out of bounds".into())),
        (Value::Tuple(values), Value::Int(index)) => usize::try_from(index)
            .ok()
            .and_then(|index| values.get(index).cloned())
            .ok_or_else(|| CompileError::Execution("tuple index out of bounds".into())),
        (Value::Map(entries), key) => entries
            .borrow()
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .map(|(_, value)| value.clone())
            .ok_or_else(|| CompileError::Execution("map key not found".into())),
        _ => Err(CompileError::Execution("value is not indexable".into())),
    }
}

fn match_pattern(
    program: &Program,
    value: &Value,
    pattern: &MatchPattern,
    bindings: &mut HashMap<String, Value>,
) -> bool {
    match pattern {
        MatchPattern::Wildcard => true,
        MatchPattern::Bind(name) => {
            bindings.insert(name.clone(), value.clone());
            true
        }
        MatchPattern::Integer(expected) => value == &Value::Int(*expected),
        MatchPattern::Float(expected) => value == &Value::Float(*expected),
        MatchPattern::Boolean(expected) => value == &Value::Bool(*expected),
        MatchPattern::String(expected) => value == &Value::String(expected.clone()),
        MatchPattern::Constructor { name, fields } => {
            let values = match value {
                Value::Tuple(values) if name == "tuple" => values.clone(),
                Value::Variant {
                    name: actual,
                    fields: values,
                } if actual == name => values.clone(),
                Value::Object(object) if object.borrow().class == *name => {
                    let object = object.borrow();
                    let Some(class) = program.classes.iter().find(|class| class.name == *name)
                    else {
                        return false;
                    };
                    class
                        .fields
                        .iter()
                        .filter_map(|field| object.fields.get(field).cloned())
                        .collect()
                }
                _ => return false,
            };
            fields.len() == values.len()
                && fields
                    .iter()
                    .zip(&values)
                    .all(|(pattern, value)| match_pattern(program, value, pattern, bindings))
        }
    }
}

fn display_value(value: &Value) -> String {
    match value {
        Value::Int(value) => value.to_string(),
        Value::Float(value) => f64::from_bits(*value).to_string(),
        Value::Bool(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Function(name) => format!("<function {name}>"),
        Value::List(values) => format_collection("[", "]", &values.borrow()),
        Value::Tuple(values) => format_collection("(", ")", values),
        Value::Set(values) => format_collection("{", "}", values),
        Value::Map(entries) => {
            let values = entries
                .borrow()
                .iter()
                .map(|(key, value)| format!("{}: {}", display_nested(key), display_nested(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{values}}}")
        }
        Value::Object(object) => {
            let object = object.borrow();
            let fields = object
                .fields
                .iter()
                .map(|(name, value)| format!("{name}={}", display_nested(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({fields})", object.class)
        }
        Value::Variant { name, fields } => {
            if fields.is_empty() {
                name.clone()
            } else {
                format!(
                    "{name}({})",
                    fields
                        .iter()
                        .map(display_nested)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        Value::Task(_) => "<task>".into(),
        Value::Channel(_) => "<channel>".into(),
        Value::Unit => "unit".into(),
    }
}

fn format_collection(open: &str, close: &str, values: &[Value]) -> String {
    let values = values
        .iter()
        .map(display_nested)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{open}{values}{close}")
}

fn display_nested(value: &Value) -> String {
    match value {
        Value::String(value) => format!("\"{value}\""),
        value => display_value(value),
    }
}

fn find_tool(candidates: &[&str]) -> Option<PathBuf> {
    for candidate in candidates {
        let path = Path::new(candidate);
        if path.components().count() > 1 && path.is_file() {
            return Some(path.into());
        }

        if let Some(paths) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&paths) {
                let executable = directory.join(candidate);
                if executable.is_file() {
                    return Some(executable);
                }
            }
        }
    }
    None
}

fn run_tool(tool: PathBuf, args: &[&Path]) -> Result<(), CompileError> {
    let output = Command::new(&tool).args(args).output()?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(CompileError::Io(std::io::Error::other(format!(
        "{} failed: {stderr}",
        tool.display()
    ))))
}

fn missing_tool(name: &str) -> CompileError {
    CompileError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("required tool `{name}` was not found"),
    ))
}
