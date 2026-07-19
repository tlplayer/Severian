#![forbid(unsafe_code)]

use severian_ast::Span;
use severian_hir::{BinaryOp, Expression, Instruction, Program};
use severian_mlir::Module;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

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
            execute_instructions(
                program,
                &test.instructions,
                &mut HashMap::new(),
                &mut output,
            )
            .map_err(|error| CompileError::Execution(format!("test `{label}` failed: {error}")))?;
            report(&format!("test {label} ... ok"));
            passed += 1;
        }
    }
    Ok(passed)
}

fn ignore_output(_: &str) {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    Unit,
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
    let mut variables = function
        .params
        .iter()
        .zip(args)
        .map(|(param, value)| (param.name.clone(), value))
        .collect();
    Ok(
        execute_instructions(program, &function.instructions, &mut variables, write_line)?
            .unwrap_or(Value::Unit),
    )
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
            Instruction::Evaluate(expression) => {
                evaluate(program, expression, variables, write_line)?;
            }
        }
    }
    Ok(None)
}

fn evaluate(
    program: &Program,
    expression: &Expression,
    variables: &HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    match expression {
        Expression::Integer(value) => Ok(Value::Int(*value)),
        Expression::Boolean(value) => Ok(Value::Bool(*value)),
        Expression::String(value) => Ok(Value::String(value.clone())),
        Expression::Variable(name) => variables
            .get(name)
            .cloned()
            .ok_or_else(|| CompileError::Execution(format!("unknown binding `{name}`"))),
        Expression::Call { function, args } => {
            let args = args
                .iter()
                .map(|arg| evaluate(program, arg, variables, write_line))
                .collect::<Result<Vec<_>, _>>()?;
            execute_function(program, function, args, write_line)
        }
        Expression::Binary { left, op, right } => {
            let left = evaluate(program, left, variables, write_line)?;
            let right = evaluate(program, right, variables, write_line)?;
            evaluate_binary(left, *op, right)
        }
    }
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
        _ => Err(CompileError::Execution("invalid binary operation".into())),
    }
}

fn display_value(value: &Value) -> String {
    match value {
        Value::Int(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Unit => "unit".into(),
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
