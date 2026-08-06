#![forbid(unsafe_code)]

use severian_ast::{Module as AstModule, Span};
use severian_hir::{
    AssignmentOp, BinaryOp, ChaosAction, Expression, Function, Instruction, MatchPattern, Program,
    Test, TestMode, UnaryOp,
};
use severian_mlir::Module;
use severian_package::PackageInterface;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
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
    Optimization(String),
    Package(String),
    Execution(String),
    ChaosThrow(String),
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
            CompileError::Optimization(message) => {
                write!(formatter, "optimization error: {message}")
            }
            CompileError::Package(message) => write!(formatter, "package error: {message}"),
            CompileError::Execution(message) => write!(formatter, "execution error: {message}"),
            CompileError::ChaosThrow(message) => write!(formatter, "injected throw: {message}"),
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
    let ast = parse_source(source)?;
    compile_ast(&ast, &[])
}

fn parse_source(source: &str) -> Result<AstModule, CompileError> {
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
    Ok(ast)
}

fn compile_ast(
    ast: &AstModule,
    interfaces: &[PackageInterface],
) -> Result<Compilation, CompileError> {
    let hir = severian_semantic::analyze_with_packages(ast, interfaces).map_err(|error| {
        CompileError::Frontend {
            stage: "semantic",
            span: error.span,
            message: error.message,
        }
    })?;
    severian_ownership::check(&hir).map_err(|error| CompileError::Ownership(error.message))?;
    let mut optimized_hir = hir.clone();
    let fusion_rules = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.fusion_rules.iter().cloned());
    let fusion_aliases = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.fusion_aliases.iter().cloned());
    let graph_rules = interfaces
        .iter()
        .flat_map(|interface| interface.compiler.graph_rules.iter().cloned());
    severian_passes::standard_pipeline_with_graph(fusion_rules, fusion_aliases, graph_rules)
        .run(&mut optimized_hir)
        .map_err(|error| CompileError::Optimization(error.to_string()))?;
    let mlir = severian_lowering::lower(&optimized_hir);

    Ok(Compilation { hir, mlir })
}

pub fn compile_path(path: &Path) -> Result<Compilation, CompileError> {
    let source = std::fs::read_to_string(path)?;
    let Some(manifest_path) = severian_package::find_manifest(path) else {
        let ast = parse_source(&source)?;
        let interfaces = load_official_interfaces(&ast)?;
        return compile_ast(&ast, &interfaces);
    };
    let dependency_sources = severian_package::load_path_dependency_sources(&manifest_path)
        .map_err(|error| CompileError::Package(error.to_string()))?;
    let mut package_source = dependency_sources.join("\n");
    if !package_source.is_empty() {
        package_source.push('\n');
    }
    package_source.push_str(&source);
    let ast = parse_source(&package_source)?;
    let interfaces = load_official_interfaces(&ast)?;
    compile_ast(&ast, &interfaces)
}

fn load_official_interfaces(module: &AstModule) -> Result<Vec<PackageInterface>, CompileError> {
    let library_root = std::env::var_os("SEVERIAN_LIBRARY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library"));
    severian_package::load_official_interfaces(module, &library_root)
        .map_err(|error| CompileError::Package(error.to_string()))
}

pub fn compile_native(compilation: &Compilation, output: &Path) -> Result<(), CompileError> {
    severian_backend::compile_native(&compilation.hir, &compilation.mlir, output)
        .map_err(|error| CompileError::Io(std::io::Error::other(error.to_string())))
}

pub fn compile_rocm(
    compilation: &Compilation,
    output: &Path,
    chip: &str,
) -> Result<(), CompileError> {
    severian_backend::compile_rocm(&compilation.hir, &compilation.mlir, output, chip)
        .map_err(|error| CompileError::Io(std::io::Error::other(error.to_string())))
}

pub fn lower_to_rocdl(compilation: &Compilation, chip: &str) -> Result<Module, CompileError> {
    severian_backend::lower_to_rocdl(&compilation.mlir, chip)
        .map_err(|error| CompileError::Io(std::io::Error::other(error.to_string())))
}

pub fn detect_amd_gpu_chip() -> Option<String> {
    severian_backend::detect_amd_gpu_chip()
}

/// Build a native executable whose entry point runs every non-integration test.
///
/// This is deliberately unavailable for a test-free library: emitting a no-op
/// entry point would make native acceptance appear to cover code it never ran.
pub fn compile_native_tests(
    compilation: &Compilation,
    output: &Path,
) -> Result<usize, CompileError> {
    let (native, count) = native_test_compilation(compilation)?;
    compile_native(&native, output)?;
    Ok(count)
}

pub fn native_test_compilation(
    compilation: &Compilation,
) -> Result<(Compilation, usize), CompileError> {
    let mut instructions = Vec::new();
    let mut count = 0;
    for function in &compilation.hir.functions {
        for test in &function.tests {
            if !test.modes.contains(&TestMode::Integration) {
                if test.modes.contains(&TestMode::Chaos) {
                    let inherited = reachable_dependencies(&compilation.hir, function)
                        .into_iter()
                        .flat_map(|dependency| &dependency.tests)
                        .map(|dependency_test| {
                            let mut rules = Vec::new();
                            collect_chaos_rules(&dependency_test.instructions, &mut rules);
                            rules.len()
                        })
                        .sum::<usize>();
                    instructions.push(Instruction::Let {
                        name: "chaos".into(),
                        value: Expression::List(
                            (0..inherited).map(|_| Expression::Integer(0)).collect(),
                        ),
                    });
                }
                instructions.extend(test.instructions.clone());
                count += 1;
            }
        }
    }
    for class in &compilation.hir.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            for test in &function.tests {
                if !test.modes.contains(&TestMode::Integration) {
                    instructions.extend(test.instructions.clone());
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return Err(CompileError::Execution(
            "source has neither `main()` nor native tests; refusing to generate a no-op executable"
                .into(),
        ));
    }
    instructions.push(Instruction::Print(Expression::String(format!(
        "{count} passed"
    ))));
    let mut hir = compilation.hir.clone();
    hir.functions.retain(|function| function.name != "main");
    hir.functions.push(Function {
        name: "main".into(),
        native_symbol: None,
        decorators: Vec::new(),
        contract: None,
        params: Vec::new(),
        return_type: severian_hir::ValueType::Unit,
        instructions,
        tests: Vec::new(),
    });
    let native = Compilation {
        mlir: severian_lowering::lower(&hir),
        hir,
    };
    Ok((native, count))
}

pub fn run(program: &Program, mut write_line: impl FnMut(&str)) -> Result<(), CompileError> {
    program.main().ok_or_else(|| CompileError::Frontend {
        stage: "semantic",
        span: Span::default(),
        message: "program must define `main`".into(),
    })?;
    execute_function(program, "main", Vec::new(), None, &mut write_line)?;
    Ok(())
}

pub fn run_tests(program: &Program, mut report: impl FnMut(&str)) -> Result<usize, CompileError> {
    let mut passed = 0;
    for function in &program.functions {
        for (index, test) in function.tests.iter().enumerate() {
            if test.modes.contains(&TestMode::Integration) {
                continue;
            }
            let label = test.name.clone().unwrap_or_else(|| {
                if function.tests.len() == 1 {
                    function.name.clone()
                } else {
                    format!("{} #{}", function.name, index + 1)
                }
            });
            let mut output = ignore_output;
            let mut variables = test_variables(program, function, test, &mut output)?;
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
                if test.modes.contains(&TestMode::Integration) {
                    continue;
                }
                let label = test.name.clone().unwrap_or_else(|| {
                    if function.tests.len() == 1 {
                        format!("{}.{}", class.name, function.name)
                    } else {
                        format!("{}.{} #{}", class.name, function.name, index + 1)
                    }
                });
                let mut output = ignore_output;
                let mut variables = test_variables(program, function, test, &mut output)?;
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

pub fn run_integration_tests(
    compilation: &Compilation,
    mut report: impl FnMut(&str),
) -> Result<usize, CompileError> {
    let has_integration_tests = compilation.hir.functions.iter().any(|function| {
        function
            .tests
            .iter()
            .any(|test| test.modes.contains(&TestMode::Integration))
    });
    if !has_integration_tests {
        return Ok(0);
    }
    let executable = std::env::temp_dir().join(format!(
        "severian-integration-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos()
    ));
    compile_native(compilation, &executable)?;
    let result = run_integration_tests_with_executable(compilation, &executable, &mut report);
    let _ = std::fs::remove_file(executable);
    result
}

fn run_integration_tests_with_executable(
    compilation: &Compilation,
    executable: &Path,
    report: &mut dyn FnMut(&str),
) -> Result<usize, CompileError> {
    let mut passed = 0;
    for function in &compilation.hir.functions {
        for (index, test) in function.tests.iter().enumerate() {
            if !test.modes.contains(&TestMode::Integration) {
                continue;
            }
            let label = test.name.clone().unwrap_or_else(|| {
                if function.tests.len() == 1 {
                    function.name.clone()
                } else {
                    format!("{} #{}", function.name, index + 1)
                }
            });
            let output = Command::new("timeout").arg("5").arg(executable).output()?;
            if !output.status.success() {
                return Err(CompileError::Execution(format!(
                    "integration test `{label}` native program exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
            let mut ignored = ignore_output;
            let mut variables = test_variables(&compilation.hir, function, test, &mut ignored)?;
            variables.insert(
                "stdout".into(),
                Value::String(String::from_utf8_lossy(&output.stdout).into_owned()),
            );
            variables.insert(
                "stderr".into(),
                Value::String(String::from_utf8_lossy(&output.stderr).into_owned()),
            );
            let assertions = test
                .instructions
                .iter()
                .filter(|instruction| !is_direct_main_call(instruction))
                .cloned()
                .collect::<Vec<_>>();
            execute_instructions(&compilation.hir, &assertions, &mut variables, &mut ignored)
                .map_err(|error| {
                    CompileError::Execution(format!("integration test `{label}` failed: {error}"))
                })?;
            report(&format!("test with integration {label} ... ok"));
            passed += 1;
        }
    }
    Ok(passed)
}

fn is_direct_main_call(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Evaluate(Expression::Call { function, args })
            if function == "main" && args.is_empty()
    )
}

fn ignore_output(_: &str) {}

fn test_variables(
    program: &Program,
    function: &Function,
    test: &Test,
    write_line: &mut dyn FnMut(&str),
) -> Result<HashMap<String, Value>, CompileError> {
    let mut variables = initial_variables(program, write_line)?;
    let mut events = Vec::new();
    if test.modes.contains(&TestMode::Chaos) {
        for dependency in reachable_dependencies(program, function) {
            for dependency_test in &dependency.tests {
                let mut rules = Vec::new();
                collect_chaos_rules(&dependency_test.instructions, &mut rules);
                for rule in rules {
                    events.push(evaluate(program, &rule, &variables, write_line)?);
                }
            }
        }
    }
    variables.insert("chaos".into(), Value::List(Rc::new(RefCell::new(events))));
    Ok(variables)
}

fn reachable_dependencies<'program>(
    program: &'program Program,
    root: &Function,
) -> Vec<&'program Function> {
    let mut pending = Vec::new();
    collect_called_functions(&root.instructions, &mut pending);
    let mut visited = HashSet::new();
    let mut dependencies = Vec::new();
    while let Some(name) = pending.pop() {
        if name == root.name || !visited.insert(name.clone()) {
            continue;
        }
        let Some(function) = program
            .functions
            .iter()
            .find(|function| function.name == name)
        else {
            continue;
        };
        collect_called_functions(&function.instructions, &mut pending);
        dependencies.push(function);
    }
    dependencies
}

fn collect_called_functions(instructions: &[Instruction], calls: &mut Vec<String>) {
    walk_instructions(instructions, &mut |expression| {
        if let Expression::Call { function, .. } = expression {
            calls.push(function.clone());
        }
    });
}

fn collect_chaos_rules(instructions: &[Instruction], rules: &mut Vec<Expression>) {
    walk_instructions(instructions, &mut |expression| {
        if matches!(expression, Expression::ChaosRule { .. }) {
            rules.push(expression.clone());
        }
    });
}

fn walk_instructions<'expression>(
    instructions: &'expression [Instruction],
    visit: &mut impl FnMut(&'expression Expression),
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Assign { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => walk_expression(value, visit),
            Instruction::Return(Some(value)) => walk_expression(value, visit),
            Instruction::Return(None) => {}
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                walk_expression(condition, visit);
                walk_instructions(then_instructions, visit);
                walk_instructions(else_instructions, visit);
            }
            Instruction::While {
                setup,
                condition,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    walk_instructions(std::slice::from_ref(setup), visit);
                }
                walk_expression(condition, visit);
                walk_instructions(instructions, visit);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    walk_instructions(std::slice::from_ref(setup), visit);
                }
                walk_expression(iterable, visit);
                walk_instructions(instructions, visit);
            }
            Instruction::Switch { value, arms } => {
                walk_expression(value, visit);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        walk_expression(source, visit);
                    }
                    if let Some(guard) = &arm.guard {
                        walk_expression(guard, visit);
                    }
                    walk_instructions(&arm.instructions, visit);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    walk_expression(channel, visit);
                }
                if let Some(setup) = setup {
                    walk_instructions(std::slice::from_ref(setup), visit);
                }
                if let Some(condition) = repeat_condition {
                    walk_expression(condition, visit);
                }
                for arm in arms {
                    if let Some(source) = &arm.source {
                        walk_expression(source, visit);
                    }
                    if let Some(guard) = &arm.guard {
                        walk_expression(guard, visit);
                    }
                    walk_instructions(&arm.instructions, visit);
                }
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    walk_expression(resource, visit);
                }
                walk_instructions(instructions, visit);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn walk_expression<'expression>(
    expression: &'expression Expression,
    visit: &mut impl FnMut(&'expression Expression),
) {
    visit(expression);
    match expression {
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                walk_expression(value, visit);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                walk_expression(key, visit);
                walk_expression(value, visit);
            }
        }
        Expression::Index { object, index } => {
            walk_expression(object, visit);
            walk_expression(index, visit);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            walk_expression(object, visit);
            for bound in [start, end, step].into_iter().flatten() {
                walk_expression(bound, visit);
            }
        }
        Expression::Member { object, .. }
        | Expression::Unary {
            expression: object, ..
        }
        | Expression::Await(object)
        | Expression::Channel(object)
        | Expression::ChaosRule { value: object, .. } => walk_expression(object, visit),
        Expression::Task { value, .. } => walk_expression(value, visit),
        Expression::MethodCall { object, args, .. } => {
            walk_expression(object, visit);
            for arg in args {
                walk_expression(arg, visit);
            }
        }
        Expression::Send { value, channel } => {
            walk_expression(value, visit);
            walk_expression(channel, visit);
        }
        Expression::ListComprehension { element, clauses } => {
            walk_expression(element, visit);
            for clause in clauses {
                walk_expression(&clause.iterable, visit);
                if let Some(condition) = &clause.condition {
                    walk_expression(condition, visit);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            walk_expression(element, visit);
            for clause in clauses {
                walk_expression(&clause.iterable, visit);
                if let Some(condition) = &clause.condition {
                    walk_expression(condition, visit);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            walk_expression(key, visit);
            walk_expression(value, visit);
            for clause in clauses {
                walk_expression(&clause.iterable, visit);
                if let Some(condition) = &clause.condition {
                    walk_expression(condition, visit);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            walk_expression(condition, visit);
            walk_expression(then_expression, visit);
            walk_expression(else_expression, visit);
        }
        Expression::FusedPipeline { input, .. } => walk_expression(input, visit),
        Expression::Ownership { value, .. } => walk_expression(value, visit),
        Expression::Lambda { body, .. } => walk_expression(body, visit),
        Expression::Call { args, .. } => {
            for arg in args {
                walk_expression(arg, visit);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            walk_expression(callee, visit);
            for arg in args {
                walk_expression(arg, visit);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                walk_expression(arg, visit);
            }
        }
        Expression::Binary { left, right, .. } => {
            walk_expression(left, visit);
            walk_expression(right, visit);
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ControlledDatabase {
    rows: Vec<Vec<String>>,
    transaction: Option<Vec<Vec<String>>>,
}

thread_local! {
    static CONTROLLED_DATABASES: RefCell<HashMap<String, Rc<RefCell<ControlledDatabase>>>> = RefCell::new(HashMap::new());
    static CONTROLLED_DATABASE_SERVERS: RefCell<HashMap<String, Rc<RefCell<ControlledDatabase>>>> = RefCell::new(HashMap::new());
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Int(i64),
    Float(u64),
    Bool(bool),
    String(String),
    Function(String),
    Lambda {
        params: Vec<String>,
        body: Box<Expression>,
        captures: HashMap<String, Value>,
    },
    Matrix {
        rows: i64,
        columns: i64,
        fill: u64,
    },
    Tensor {
        shape: Vec<i64>,
        values: Vec<u64>,
    },
    List(Rc<RefCell<Vec<Value>>>),
    Tuple(Vec<Value>),
    Map(Rc<RefCell<Vec<(Value, Value)>>>),
    Set(Vec<Value>),
    Object(Rc<RefCell<ObjectValue>>),
    Database(Rc<RefCell<ControlledDatabase>>),
    DatabaseServer {
        address: String,
        database: Rc<RefCell<ControlledDatabase>>,
    },
    DatabaseConnection(Rc<RefCell<ControlledDatabase>>),
    Variant {
        name: String,
        fields: Vec<Value>,
    },
    Task(Box<Value>),
    Channel(Rc<RefCell<VecDeque<Value>>>),
    ChaosRule {
        function: String,
        action: ChaosAction,
        value: Box<Value>,
        hit: Rc<Cell<bool>>,
    },
    Unit,
    Break,
    Continue,
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
    chaos_event: Option<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    let function = program
        .functions
        .iter()
        .find(|function| function.name == name)
        .ok_or_else(|| CompileError::Execution(format!("unknown function `{name}`")))?;
    let mut variables = initial_variables(program, write_line)?;
    if let Some(event) = chaos_event {
        variables.insert("__chaos_event".into(), event);
    }
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

fn invoke_callable(
    program: &Program,
    callable: Value,
    args: Vec<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    match callable {
        Value::Function(function) => execute_function(program, &function, args, None, write_line),
        Value::Lambda {
            params,
            body,
            mut captures,
        } => {
            if params.len() != args.len() {
                return Err(CompileError::Execution(format!(
                    "lambda expects {} arguments",
                    params.len()
                )));
            }
            captures.extend(params.into_iter().zip(args));
            evaluate(program, &body, &captures, write_line)
        }
        _ => Err(CompileError::Execution("value is not callable".into())),
    }
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
            Instruction::Break => return Ok(Some(Value::Break)),
            Instruction::Continue => return Ok(Some(Value::Continue)),
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
                ..
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
                        match value {
                            Value::Break => break,
                            Value::Continue => continue,
                            value => return Ok(Some(value)),
                        }
                    }
                }
            }
            Instruction::For {
                setup,
                pattern,
                iterable,
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
                let values = iterable_values(evaluate(program, iterable, variables, write_line)?)?;
                for value in values {
                    let active_hit = if let Value::ChaosRule { hit, .. } = &value {
                        hit.set(false);
                        variables.insert("__chaos_event".into(), value.clone());
                        Some(hit.clone())
                    } else {
                        None
                    };
                    let mut bindings = HashMap::new();
                    if !match_pattern(program, &value, pattern, &mut bindings) {
                        return Err(CompileError::Execution("for pattern did not match".into()));
                    }
                    variables.extend(bindings);
                    let result: Result<(), CompileError> =
                        match execute_instructions(program, instructions, variables, write_line) {
                            Ok(Some(Value::Break)) => break,
                            Ok(Some(Value::Continue)) => continue,
                            Ok(Some(value)) => return Ok(Some(value)),
                            Ok(None) => Ok(()),
                            Err(CompileError::ChaosThrow(_))
                                if matches!(
                                    variables.get("__chaos_event"),
                                    Some(Value::ChaosRule {
                                        action: ChaosAction::Throw,
                                        ..
                                    })
                                ) =>
                            {
                                Ok(())
                            }
                            Err(error) => return Err(error),
                        };
                    result?;
                    if active_hit.as_ref().is_some_and(|hit| !hit.get()) {
                        return Err(CompileError::Execution(
                            "chaos event did not reach its target function".into(),
                        ));
                    }
                    if active_hit.is_some() {
                        variables.remove("__chaos_event");
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
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                if let Some(setup) = setup {
                    execute_instructions(
                        program,
                        std::slice::from_ref(setup),
                        variables,
                        write_line,
                    )?;
                }
                loop {
                    if let Some(condition) = repeat_condition {
                        if !truthy(&evaluate(program, condition, variables, write_line)?)? {
                            break;
                        }
                    }

                    let mut selected = None;
                    for channel_expression in channels {
                        let Value::Channel(channel) =
                            evaluate(program, channel_expression, variables, write_line)?
                        else {
                            return Err(CompileError::Execution(
                                "channel switch source is not a channel".into(),
                            ));
                        };
                        let received = channel.borrow_mut().pop_front();
                        if let Some(value) = received {
                            selected = arms
                                .iter()
                                .find(|arm| arm.source.as_ref() == Some(channel_expression))
                                .map(|arm| (arm, value));
                            break;
                        }
                    }

                    let (arm, value) = if let Some(selected) = selected {
                        selected
                    } else if let Some(arm) = arms.iter().find(|arm| arm.source.is_none()) {
                        (
                            arm,
                            Value::Variant {
                                name: "fail".into(),
                                fields: vec![Value::String("all channels are empty".into())],
                            },
                        )
                    } else {
                        return Err(CompileError::Execution(
                            "all selected channels are empty".into(),
                        ));
                    };

                    let mut bindings = HashMap::new();
                    if !match_pattern(program, &value, &arm.pattern, &mut bindings) {
                        return Err(CompileError::Execution(
                            "channel switch pattern did not match".into(),
                        ));
                    }
                    let bound_names = bindings.keys().cloned().collect::<Vec<_>>();
                    variables.extend(bindings);
                    if let Some(guard) = &arm.guard {
                        if !truthy(&evaluate(program, guard, variables, write_line)?)? {
                            return Err(CompileError::Execution(
                                "channel switch guard rejected the received value".into(),
                            ));
                        }
                    }
                    if let Some(value) =
                        execute_instructions(program, &arm.instructions, variables, write_line)?
                    {
                        return Ok(Some(value));
                    }
                    for name in bound_names {
                        variables.remove(&name);
                    }

                    if repeat_condition.is_none() {
                        break;
                    }
                }
            }
            Instruction::Evaluate(expression) => {
                evaluate(program, expression, variables, write_line)?;
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    evaluate(program, resource, variables, write_line)?;
                }
                if let Some(value) =
                    execute_instructions(program, instructions, variables, write_line)?
                {
                    return Ok(Some(value));
                }
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
                    let mut values = values.borrow_mut();
                    let index = normalize_index(index, values.len(), "list")?;
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
        Expression::Task { value, .. } => Ok(Value::Task(Box::new(evaluate(
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
        Expression::ChaosRule {
            function,
            action,
            value,
        } => Ok(Value::ChaosRule {
            function: function.clone(),
            action: *action,
            value: Box::new(evaluate(program, value, variables, write_line)?),
            hit: Rc::new(Cell::new(false)),
        }),
        Expression::Variable(name) => variables
            .get(name)
            .cloned()
            .ok_or_else(|| CompileError::Execution(format!("unknown binding `{name}`"))),
        Expression::Lambda { params, body } => Ok(Value::Lambda {
            params: params.clone(),
            body: body.clone(),
            captures: variables.clone(),
        }),
        Expression::Ownership { value, .. } => evaluate(program, value, variables, write_line),
        Expression::Index { object, index } => {
            let object = evaluate(program, object, variables, write_line)?;
            let index = evaluate(program, index, variables, write_line)?;
            index_value(object, index)
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            let object = evaluate(program, object, variables, write_line)?;
            let mut evaluate_bound =
                |bound: &Option<Box<Expression>>| -> Result<Option<i64>, CompileError> {
                    bound
                        .as_ref()
                        .map(
                            |bound| match evaluate(program, bound, variables, write_line)? {
                                Value::Int(value) => Ok(value),
                                _ => Err(CompileError::Execution(
                                    "slice bounds must be integers".into(),
                                )),
                            },
                        )
                        .transpose()
                };
            slice_value(
                object,
                evaluate_bound(start)?,
                evaluate_bound(end)?,
                evaluate_bound(step)?,
            )
        }
        Expression::Format { template, .. } => {
            let mut formatted = template.clone();
            for (name, value) in variables {
                formatted = formatted.replace(&format!("{{{name}}}"), &display_value(value));
            }
            Ok(Value::String(formatted))
        }
        Expression::ListComprehension { element, clauses } => {
            let mut result = Vec::new();
            for inner in comprehension_bindings(program, clauses, variables, write_line)? {
                result.push(evaluate(program, element, &inner, write_line)?);
            }
            Ok(Value::List(Rc::new(RefCell::new(result))))
        }
        Expression::SetComprehension { element, clauses } => {
            let mut result = Vec::new();
            for inner in comprehension_bindings(program, clauses, variables, write_line)? {
                let value = evaluate(program, element, &inner, write_line)?;
                if !result.contains(&value) {
                    result.push(value);
                }
            }
            Ok(Value::Set(result))
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            let mut result = Vec::new();
            for inner in comprehension_bindings(program, clauses, variables, write_line)? {
                let key = evaluate(program, key, &inner, write_line)?;
                let value = evaluate(program, value, &inner, write_line)?;
                if let Some((_, existing)) = result
                    .iter_mut()
                    .find(|(candidate, _): &&mut (Value, Value)| candidate == &key)
                {
                    *existing = value;
                } else {
                    result.push((key, value));
                }
            }
            Ok(Value::Map(Rc::new(RefCell::new(result))))
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
            let chaos_event = variables.get("__chaos_event").cloned();
            if let Some(Value::ChaosRule {
                function: target,
                action,
                value,
                hit,
            }) = &chaos_event
            {
                if target == function {
                    hit.set(true);
                    return match action {
                        ChaosAction::Return => Ok((**value).clone()),
                        ChaosAction::Throw => Err(CompileError::ChaosThrow(display_value(value))),
                    };
                }
            }
            execute_call(program, function, args, chaos_event, write_line)
        }
        Expression::CallValue { callee, args, .. } => {
            let callable = evaluate(program, callee, variables, write_line)?;
            let args = evaluate_all(program, args, variables, write_line)?;
            match callable {
                Value::Function(function) => execute_function(
                    program,
                    &function,
                    args,
                    variables.get("__chaos_event").cloned(),
                    write_line,
                ),
                Value::Lambda {
                    params,
                    body,
                    mut captures,
                } => {
                    if params.len() != args.len() {
                        return Err(CompileError::Execution(format!(
                            "lambda expects {} arguments",
                            params.len()
                        )));
                    }
                    captures.extend(params.into_iter().zip(args));
                    evaluate(program, &body, &captures, write_line)
                }
                _ => Err(CompileError::Execution("value is not callable".into())),
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => match evaluate(program, condition, variables, write_line)? {
            Value::Bool(true) => evaluate(program, then_expression, variables, write_line),
            Value::Bool(false) => evaluate(program, else_expression, variables, write_line),
            _ => Err(CompileError::Execution(
                "conditional expression requires a boolean condition".into(),
            )),
        },
        Expression::FusedPipeline { .. } => Err(CompileError::Execution(
            "backend-only fused pipeline reached the controlled evaluator".into(),
        )),
        Expression::Binary { left, op, right } => {
            let left = evaluate(program, left, variables, write_line)?;
            if matches!((op, &left), (BinaryOp::And, Value::Bool(false))) {
                return Ok(Value::Bool(false));
            }
            if matches!((op, &left), (BinaryOp::Or, Value::Bool(true))) {
                return Ok(Value::Bool(true));
            }
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
    chaos_event: Option<Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Value, CompileError> {
    let linked_function = function
        .rsplit_once('.')
        .map(|(_, name)| name)
        .filter(|name| {
            program
                .functions
                .iter()
                .any(|candidate| candidate.name == *name)
        })
        .unwrap_or(function);
    let linked_definition = program
        .functions
        .iter()
        .find(|candidate| candidate.name == linked_function);
    if linked_definition.is_some_and(|function| function.native_symbol.is_none()) {
        return execute_function(program, linked_function, args, chaos_event, write_line);
    }
    let function = linked_definition
        .and_then(|function| function.native_symbol.as_deref())
        .map(|symbol| match symbol {
            "__sev_file_read" => "platform.fileRead",
            "__sev_file_write" => "platform.fileWrite",
            "__sev_json_decode" => "platform.jsonDecode",
            "__sev_json_encode" => "platform.jsonEncode",
            "__sev_log_info" => "platform.logInfo",
            "__sev_log_error" => "platform.logError",
            "__sev_network_listen" => "platform.networkListen",
            "__sev_network_loopback_echo" => "platform.networkLoopbackEcho",
            "__sev_regex_matches" => "platform.regexMatches",
            "__sev_host_container_backend" => "platform.hostContainerBackend",
            "__sev_host_kvm_api_version" => "platform.hostKvmApiVersion",
            "__sev_host_kvm_create_probe" => "platform.hostKvmCreateProbe",
            "__sev_host_page_size" => "platform.hostPageSize",
            "__sev_database_open" => "platform.databaseOpen",
            "__sev_database_execute" => "platform.databaseExecute",
            "__sev_database_query" => "platform.databaseQuery",
            "__sev_database_close" => "platform.databaseClose",
            "__sev_database_server_start" => "platform.databaseServerStart",
            "__sev_database_server_address" => "platform.databaseServerAddress",
            "__sev_database_server_connect" => "platform.databaseServerConnect",
            "__sev_database_server_execute" => "platform.databaseServerExecute",
            "__sev_database_server_query" => "platform.databaseServerQuery",
            "__sev_database_server_close" => "platform.databaseServerClose",
            "__sev_database_server_stop" => "platform.databaseServerStop",
            "__sev_tensor_from_list" => "platform.tensorFromList",
            "__sev_tensor_to_list" => "platform.tensorToList",
            "__sev_tensor_shape" => "platform.tensorShape",
            "__sev_tensor_relu" => "platform.tensorRelu",
            "__sev_tensor_add" => "platform.tensorAdd",
            "__sev_tensor_matmul" => "platform.tensorMatmul",
            _ => function,
        })
        .unwrap_or(function);
    match function {
        "print" => {
            write_line(&display_value(args.first().unwrap_or(&Value::Unit)));
            Ok(Value::Unit)
        }
        "panic" => Err(CompileError::Execution(
            args.iter().map(display_value).collect::<Vec<_>>().join(" "),
        )),
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
        "string" => match args.as_slice() {
            [value] => Ok(Value::String(display_value(value))),
            _ => Err(CompileError::Execution(
                "string expects exactly one value".into(),
            )),
        },
        "range" => match args.as_slice() {
            [Value::Int(end)] => Ok(Value::List(Rc::new(RefCell::new(
                range_values(0, *end, 1)?
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            )))),
            [Value::Int(start), Value::Int(end)] => Ok(Value::List(Rc::new(RefCell::new(
                range_values(*start, *end, 1)?
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            )))),
            [Value::Int(start), Value::Int(end), Value::Int(step)] => {
                Ok(Value::List(Rc::new(RefCell::new(
                    range_values(*start, *end, *step)?
                        .into_iter()
                        .map(Value::Int)
                        .collect(),
                ))))
            }
            _ => Err(CompileError::Execution(
                "range expects one to three integers".into(),
            )),
        },
        "indices" => match args.as_slice() {
            [Value::List(values)] => Ok(Value::List(Rc::new(RefCell::new(
                (0..values.borrow().len())
                    .map(|index| Value::Int(index as i64))
                    .collect(),
            )))),
            _ => Err(CompileError::Execution("indices expects a list".into())),
        },
        "enumerate" => match args.as_slice() {
            [value] => Ok(Value::List(Rc::new(RefCell::new(
                iterable_values(value.clone())?
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| Value::Tuple(vec![Value::Int(index as i64), value]))
                    .collect(),
            )))),
            _ => Err(CompileError::Execution(
                "enumerate expects one iterable".into(),
            )),
        },
        "zip" => match args.as_slice() {
            [left, right] => Ok(Value::List(Rc::new(RefCell::new(
                iterable_values(left.clone())?
                    .into_iter()
                    .zip(iterable_values(right.clone())?)
                    .map(|(left, right)| Value::Tuple(vec![left, right]))
                    .collect(),
            )))),
            _ => Err(CompileError::Execution("zip expects two iterables".into())),
        },
        "any" => match args.as_slice() {
            [value] => Ok(Value::Bool(
                iterable_values(value.clone())?
                    .iter()
                    .map(truthy)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .any(|value| value),
            )),
            _ => Err(CompileError::Execution("any expects one iterable".into())),
        },
        "all" => match args.as_slice() {
            [value] => Ok(Value::Bool(
                iterable_values(value.clone())?
                    .iter()
                    .map(truthy)
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .all(|value| value),
            )),
            _ => Err(CompileError::Execution("all expects one iterable".into())),
        },
        "abs" => match args.as_slice() {
            [Value::Int(value)] => {
                Ok(Value::Int(value.checked_abs().ok_or_else(|| {
                    CompileError::Execution("absolute value overflowed".into())
                })?))
            }
            [Value::Float(value)] => Ok(Value::Float(f64::from_bits(*value).abs().to_bits())),
            _ => Err(CompileError::Execution("abs expects one number".into())),
        },
        "min" | "max" => match args.as_slice() {
            [left, right] => {
                let ordering = controlled_value_ordering(left, right);
                Ok(
                    if (function == "min" && ordering.is_le())
                        || (function == "max" && ordering.is_ge())
                    {
                        left.clone()
                    } else {
                        right.clone()
                    },
                )
            }
            _ => Err(CompileError::Execution(format!(
                "{function} expects two values"
            ))),
        },
        "divmod" => match args.as_slice() {
            [Value::Int(left), Value::Int(right)] if *right != 0 => Ok(Value::Tuple(vec![
                Value::Int(left.div_euclid(*right)),
                Value::Int(left.rem_euclid(*right)),
            ])),
            _ => Err(CompileError::Execution(
                "divmod expects two integers and a nonzero divisor".into(),
            )),
        },
        "read" => Ok(Value::Variant {
            name: "ok".into(),
            fields: vec![Value::String("settings".into())],
        }),
        "platform.fileRead" => match args.as_slice() {
            [Value::String(path)] => match std::fs::read_to_string(path) {
                Ok(contents) => Ok(Value::Variant {
                    name: "ok".into(),
                    fields: vec![Value::String(contents)],
                }),
                Err(error) => Ok(Value::Variant {
                    name: "failure".into(),
                    fields: vec![Value::String(error.to_string())],
                }),
            },
            _ => Err(CompileError::Execution(
                "platform.fileRead expects a path".into(),
            )),
        },
        "platform.fileWrite" => match args.as_slice() {
            [Value::String(path), Value::String(contents)] => {
                match std::fs::write(path, contents) {
                    Ok(()) => Ok(Value::Variant {
                        name: "ok".into(),
                        fields: Vec::new(),
                    }),
                    Err(error) => Ok(Value::Variant {
                        name: "failure".into(),
                        fields: vec![Value::String(error.to_string())],
                    }),
                }
            }
            _ => Err(CompileError::Execution(
                "platform.fileWrite expects a path and contents".into(),
            )),
        },
        "platform.jsonDecode" => match args.as_slice() {
            [Value::String(text)] => Ok(Value::Variant {
                name: "ok".into(),
                fields: vec![decode_json_value(text)?],
            }),
            _ => Err(CompileError::Execution(
                "platform.jsonDecode expects text".into(),
            )),
        },
        "platform.jsonEncode" => match args.as_slice() {
            [value] => Ok(Value::String(encode_json_value(value))),
            _ => Err(CompileError::Execution(
                "platform.jsonEncode expects one value".into(),
            )),
        },
        "platform.logInfo" => match args.as_slice() {
            [Value::String(message)] => {
                write_line(&format!("INFO {message}"));
                Ok(Value::Unit)
            }
            _ => Err(CompileError::Execution(
                "platform.logInfo expects a message".into(),
            )),
        },
        "platform.logError" => match args.as_slice() {
            [Value::String(message), _] => {
                write_line(&format!("ERROR {message}"));
                Ok(Value::Unit)
            }
            _ => Err(CompileError::Execution(
                "platform.logError expects a message and cause".into(),
            )),
        },
        "platform.networkListen" => match args.as_slice() {
            [Value::String(address)] => Ok(Value::Variant {
                name: "ok".into(),
                fields: vec![Value::String(address.clone())],
            }),
            _ => Err(CompileError::Execution(
                "platform.networkListen expects an address".into(),
            )),
        },
        "platform.networkLoopbackEcho" => match args.as_slice() {
            [Value::String(message)] => Ok(Value::Variant {
                name: "ok".into(),
                fields: vec![Value::String(message.clone())],
            }),
            _ => Err(CompileError::Execution(
                "platform.networkLoopbackEcho expects a message".into(),
            )),
        },
        "platform.hostContainerBackend" => match args.as_slice() {
            [] => Ok(Value::String("linux".into())),
            _ => Err(CompileError::Execution(
                "platform.hostContainerBackend expects no arguments".into(),
            )),
        },
        "platform.hostKvmApiVersion" => match args.as_slice() {
            [] => Ok(Value::Int(-1)),
            _ => Err(CompileError::Execution(
                "platform.hostKvmApiVersion expects no arguments".into(),
            )),
        },
        "platform.hostKvmCreateProbe" => match args.as_slice() {
            [] => Ok(Value::Bool(false)),
            _ => Err(CompileError::Execution(
                "platform.hostKvmCreateProbe expects no arguments".into(),
            )),
        },
        "platform.hostPageSize" => match args.as_slice() {
            [] => Ok(Value::Int(4096)),
            _ => Err(CompileError::Execution(
                "platform.hostPageSize expects no arguments".into(),
            )),
        },
        "platform.databaseOpen" => match args.as_slice() {
            [Value::String(path)] => {
                let database = if path == ":memory:" {
                    Rc::new(RefCell::new(ControlledDatabase::default()))
                } else {
                    CONTROLLED_DATABASES.with(|databases| {
                        databases
                            .borrow_mut()
                            .entry(path.clone())
                            .or_default()
                            .clone()
                    })
                };
                Ok(ok_value(Value::Database(database)))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseOpen expects a path".into(),
            )),
        },
        "platform.databaseExecute" => match args.as_slice() {
            [database, Value::String(statement)] => {
                let database = controlled_database_handle(database)?;
                let changed = controlled_database_execute(&mut database.borrow_mut(), statement)?;
                Ok(ok_value(Value::Int(changed)))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseExecute expects a database and SQL".into(),
            )),
        },
        "platform.databaseQuery" => match args.as_slice() {
            [database, Value::String(statement)] => {
                let database = controlled_database_handle(database)?;
                let rows = controlled_database_query(&database.borrow(), statement);
                Ok(ok_value(rows))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseQuery expects a database and SQL".into(),
            )),
        },
        "platform.databaseClose" => match args.as_slice() {
            [Value::Database(_)] => Ok(ok_value(Value::Unit)),
            _ => Err(CompileError::Execution(
                "platform.databaseClose expects a database".into(),
            )),
        },
        "platform.databaseServerStart" => match args.as_slice() {
            [Value::String(path)] => {
                let database = if path == ":memory:" {
                    Rc::new(RefCell::new(ControlledDatabase::default()))
                } else {
                    CONTROLLED_DATABASES.with(|databases| {
                        databases
                            .borrow_mut()
                            .entry(path.clone())
                            .or_default()
                            .clone()
                    })
                };
                let address = "127.0.0.1:47001".to_owned();
                CONTROLLED_DATABASE_SERVERS.with(|servers| {
                    servers
                        .borrow_mut()
                        .insert(address.clone(), database.clone());
                });
                Ok(ok_value(Value::DatabaseServer { address, database }))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseServerStart expects a path".into(),
            )),
        },
        "platform.databaseServerAddress" => match args.as_slice() {
            [Value::DatabaseServer { address, .. }] => Ok(Value::String(address.clone())),
            _ => Err(CompileError::Execution(
                "platform.databaseServerAddress expects a server".into(),
            )),
        },
        "platform.databaseServerConnect" => match args.as_slice() {
            [Value::String(address)] => CONTROLLED_DATABASE_SERVERS.with(|servers| {
                servers
                    .borrow()
                    .get(address)
                    .cloned()
                    .map(Value::DatabaseConnection)
                    .map(ok_value)
                    .ok_or_else(|| CompileError::Execution("database server is unavailable".into()))
            }),
            _ => Err(CompileError::Execution(
                "platform.databaseServerConnect expects an address".into(),
            )),
        },
        "platform.databaseServerExecute" => match args.as_slice() {
            [database, Value::String(statement)] => {
                let database = controlled_database_handle(database)?;
                let changed = controlled_database_execute(&mut database.borrow_mut(), statement)?;
                Ok(ok_value(Value::Int(changed)))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseServerExecute expects a connection and SQL".into(),
            )),
        },
        "platform.databaseServerQuery" => match args.as_slice() {
            [database, Value::String(statement)] => {
                let database = controlled_database_handle(database)?;
                let rows = controlled_database_query(&database.borrow(), statement);
                Ok(ok_value(rows))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseServerQuery expects a connection and SQL".into(),
            )),
        },
        "platform.databaseServerClose" => match args.as_slice() {
            [Value::DatabaseConnection(_)] => Ok(ok_value(Value::Unit)),
            _ => Err(CompileError::Execution(
                "platform.databaseServerClose expects a connection".into(),
            )),
        },
        "platform.databaseServerStop" => match args.as_slice() {
            [Value::DatabaseServer { address, .. }] => {
                CONTROLLED_DATABASE_SERVERS.with(|servers| {
                    servers.borrow_mut().remove(address);
                });
                Ok(ok_value(Value::Unit))
            }
            _ => Err(CompileError::Execution(
                "platform.databaseServerStop expects a server".into(),
            )),
        },
        "platform.tensorFromList" => match args.as_slice() {
            [Value::List(values), Value::List(shape)] => {
                let values = values
                    .borrow()
                    .iter()
                    .map(|value| match value {
                        Value::Float(value) => Ok(*value),
                        Value::Int(value) => Ok((*value as f64).to_bits()),
                        _ => Err(CompileError::Execution(
                            "tensor values must be numeric".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shape = shape
                    .borrow()
                    .iter()
                    .map(|value| match value {
                        Value::Int(value) => Ok(*value),
                        _ => Err(CompileError::Execution(
                            "tensor shape must contain integers".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let expected = shape.iter().try_fold(1i64, |size, axis| {
                    size.checked_mul(*axis)
                        .ok_or_else(|| CompileError::Execution("tensor shape overflows".into()))
                })?;
                if expected < 0 || expected as usize != values.len() {
                    return Err(CompileError::Execution(
                        "tensor shape does not match its values".into(),
                    ));
                }
                Ok(Value::Tensor { shape, values })
            }
            _ => Err(CompileError::Execution(
                "tensor construction expects values and shape lists".into(),
            )),
        },
        "platform.tensorToList" => match args.as_slice() {
            [Value::Tensor { values, .. }] => Ok(Value::List(Rc::new(RefCell::new(
                values.iter().copied().map(Value::Float).collect(),
            )))),
            _ => Err(CompileError::Execution("expected a tensor".into())),
        },
        "platform.tensorShape" => match args.as_slice() {
            [Value::Tensor { shape, .. }] => Ok(Value::List(Rc::new(RefCell::new(
                shape.iter().copied().map(Value::Int).collect(),
            )))),
            _ => Err(CompileError::Execution("expected a tensor".into())),
        },
        "platform.tensorRelu" => match args.as_slice() {
            [Value::Tensor { shape, values }] => Ok(Value::Tensor {
                shape: shape.clone(),
                values: values
                    .iter()
                    .map(|value| f64::from_bits(*value).max(0.0).to_bits())
                    .collect(),
            }),
            _ => Err(CompileError::Execution("ReLU expects a tensor".into())),
        },
        "platform.tensorAdd" => match args.as_slice() {
            [Value::Tensor {
                shape: left_shape,
                values: left,
            }, Value::Tensor {
                shape: right_shape,
                values: right,
            }] if left_shape == right_shape => Ok(Value::Tensor {
                shape: left_shape.clone(),
                values: left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| (f64::from_bits(*left) + f64::from_bits(*right)).to_bits())
                    .collect(),
            }),
            _ => Err(CompileError::Execution(
                "tensor addition requires equal shapes".into(),
            )),
        },
        "platform.tensorMatmul" => match args.as_slice() {
            [Value::Tensor {
                shape: left_shape,
                values: left,
            }, Value::Tensor {
                shape: right_shape,
                values: right,
            }] if left_shape.len() == 2
                && right_shape.len() == 2
                && left_shape[1] == right_shape[0] =>
            {
                let rows = left_shape[0] as usize;
                let inner = left_shape[1] as usize;
                let columns = right_shape[1] as usize;
                let mut values = vec![0.0f64.to_bits(); rows * columns];
                for row in 0..rows {
                    for column in 0..columns {
                        let mut total = 0.0;
                        for index in 0..inner {
                            total += f64::from_bits(left[row * inner + index])
                                * f64::from_bits(right[index * columns + column]);
                        }
                        values[row * columns + column] = total.to_bits();
                    }
                }
                Ok(Value::Tensor {
                    shape: vec![rows as i64, columns as i64],
                    values,
                })
            }
            _ => Err(CompileError::Execution(
                "tensor matrix dimensions do not align".into(),
            )),
        },
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
        "math.eye" | "matrix.identity" => match args.as_slice() {
            [Value::Int(size)] => Ok(Value::Matrix {
                rows: *size,
                columns: *size,
                fill: 1.0f64.to_bits(),
            }),
            _ => Err(CompileError::Execution(
                "identity expects an integer size".into(),
            )),
        },
        "math.rand" | "matrix.random" => match args.as_slice() {
            [Value::Int(rows), Value::Int(columns)] => Ok(Value::Matrix {
                rows: *rows,
                columns: *columns,
                fill: 1.0f64.to_bits(),
            }),
            _ => Err(CompileError::Execution(
                "random expects two dimensions".into(),
            )),
        },
        "math.matrixMultiply" | "matrix.multiply" => match args.as_slice() {
            [Value::Matrix {
                rows,
                columns,
                fill,
            }, Value::Matrix {
                rows: right_rows,
                columns: right_columns,
                fill: right_fill,
            }] if columns == right_rows => Ok(Value::Matrix {
                rows: *rows,
                columns: *right_columns,
                fill: (f64::from_bits(*fill) * f64::from_bits(*right_fill)).to_bits(),
            }),
            _ => Err(CompileError::Execution(
                "matrix dimensions do not align".into(),
            )),
        },
        "math.scale" | "matrix.scale" => match args.as_slice() {
            [Value::Matrix {
                rows,
                columns,
                fill,
            }, Value::Float(factor)] => Ok(Value::Matrix {
                rows: *rows,
                columns: *columns,
                fill: (f64::from_bits(*fill) * f64::from_bits(*factor)).to_bits(),
            }),
            _ => Err(CompileError::Execution(
                "matrix scale expects a matrix and float".into(),
            )),
        },
        "regex.matches" | "platform.regexMatches" => match args.as_slice() {
            [Value::String(text), Value::String(_)] => {
                let Some((prefix, suffix)) = text.split_once('-') else {
                    return Ok(Value::Bool(false));
                };
                Ok(Value::Bool(
                    !prefix.is_empty()
                        && !suffix.is_empty()
                        && prefix
                            .chars()
                            .all(|character| character.is_ascii_lowercase())
                        && suffix.chars().all(|character| character.is_ascii_digit()),
                ))
            }
            _ => Err(CompileError::Execution(
                "regex.matches expects strings".into(),
            )),
        },
        "Number.zero" => Ok(Value::Int(0)),
        _ => execute_function(program, function, args, chaos_event, write_line),
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
        fields: HashMap::new(),
    }));
    if class.constructors.is_empty() {
        if args.is_empty() && class.field_defaults.iter().all(Option::is_some) {
            for (field, default) in class.fields.iter().zip(&class.field_defaults) {
                let value = evaluate(
                    program,
                    default.as_ref().expect("all defaults were checked"),
                    &HashMap::new(),
                    write_line,
                )?;
                object.borrow_mut().fields.insert(field.clone(), value);
            }
        } else if args.len() == class.fields.len() {
            for (field, value) in class.fields.iter().zip(args) {
                object.borrow_mut().fields.insert(field.clone(), value);
            }
        } else {
            return Err(CompileError::Execution(format!(
                "`{class_name}` expects {} arguments or generated field defaults",
                class.fields.len(),
            )));
        }
    } else {
        for (field, default) in class.fields.iter().zip(&class.field_defaults) {
            let value = if let Some(default) = default {
                evaluate(program, default, &HashMap::new(), write_line)?
            } else {
                Value::Unit
            };
            object.borrow_mut().fields.insert(field.clone(), value);
        }
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
        (Value::List(values), "append" | "add") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "append expects one argument".into(),
                ));
            };
            values.borrow_mut().push(value.clone());
            return Ok(Value::Unit);
        }
        (Value::List(values), "pop") => {
            let mut values = values.borrow_mut();
            if values.is_empty() {
                return Err(CompileError::Execution("cannot pop an empty list".into()));
            }
            let index = match args.as_slice() {
                [] => values.len() - 1,
                [Value::Int(index)] => normalize_index(*index, values.len(), "list")?,
                _ => {
                    return Err(CompileError::Execution(
                        "pop expects an optional index".into(),
                    ))
                }
            };
            return Ok(values.remove(index));
        }
        (Value::List(values), "popleft") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "popleft expects no arguments".into(),
                ));
            }
            if values.borrow().is_empty() {
                return Err(CompileError::Execution("cannot pop an empty list".into()));
            }
            return Ok(values.borrow_mut().remove(0));
        }
        (Value::List(values), "appendleft") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "appendleft expects one value".into(),
                ));
            };
            values.borrow_mut().insert(0, value.clone());
            return Ok(Value::Unit);
        }
        (Value::List(values), "extend") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "extend expects one iterable".into(),
                ));
            };
            values.borrow_mut().extend(iterable_values(value.clone())?);
            return Ok(Value::Unit);
        }
        (Value::List(values), "insert") => {
            let [Value::Int(index), value] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "insert expects an index and value".into(),
                ));
            };
            let mut values = values.borrow_mut();
            let length = values.len() as i64;
            let index = if *index < 0 {
                (length + index).max(0)
            } else {
                *index
            };
            values.insert(index.min(length) as usize, value.clone());
            return Ok(Value::Unit);
        }
        (Value::List(values), "remove") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution("remove expects one value".into()));
            };
            let mut values = values.borrow_mut();
            let index = values
                .iter()
                .position(|candidate| candidate == value)
                .ok_or_else(|| CompileError::Execution("remove value was not present".into()))?;
            values.remove(index);
            return Ok(Value::Unit);
        }
        (Value::List(values), "heapPush") => {
            let [value] = args.as_slice() else {
                return Err(CompileError::Execution("heapPush expects one value".into()));
            };
            let mut values = values.borrow_mut();
            values.push(value.clone());
            let mut child = values.len() - 1;
            while child > 0 {
                let parent = (child - 1) / 2;
                if controlled_value_ordering(&values[parent], &values[child]).is_le() {
                    break;
                }
                values.swap(parent, child);
                child = parent;
            }
            return Ok(Value::Unit);
        }
        (Value::List(values), "heapPop") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "heapPop expects no arguments".into(),
                ));
            }
            let mut values = values.borrow_mut();
            if values.is_empty() {
                return Err(CompileError::Execution("cannot pop an empty heap".into()));
            }
            let last = values.pop().unwrap();
            if values.is_empty() {
                return Ok(last);
            }
            let result = std::mem::replace(&mut values[0], last);
            let mut parent = 0;
            loop {
                let left = parent * 2 + 1;
                if left >= values.len() {
                    break;
                }
                let right = left + 1;
                let child = if right < values.len()
                    && controlled_value_ordering(&values[right], &values[left]).is_lt()
                {
                    right
                } else {
                    left
                };
                if controlled_value_ordering(&values[parent], &values[child]).is_le() {
                    break;
                }
                values.swap(parent, child);
                parent = child;
            }
            return Ok(result);
        }
        (Value::List(values), "last") => {
            if !args.is_empty() {
                return Err(CompileError::Execution("last expects no arguments".into()));
            }
            return values
                .borrow()
                .last()
                .cloned()
                .ok_or_else(|| CompileError::Execution("an empty list has no last value".into()));
        }
        (Value::List(values), "reversed") => {
            let mut reversed = values.borrow().clone();
            reversed.reverse();
            return Ok(Value::List(Rc::new(RefCell::new(reversed))));
        }
        (Value::List(values), "sorted") => {
            let (key, reverse) = match args.as_slice() {
                [] => (None, false),
                [Value::Bool(reverse)] => (None, *reverse),
                [callable @ (Value::Function(_) | Value::Lambda { .. })] => (Some(callable), false),
                [callable @ (Value::Function(_) | Value::Lambda { .. }), Value::Bool(reverse)] => {
                    (Some(callable), *reverse)
                }
                _ => {
                    return Err(CompileError::Execution(
                        "sorted expects an optional key function and reverse boolean".into(),
                    ))
                }
            };
            let mut keyed = values
                .borrow()
                .iter()
                .cloned()
                .map(|value| {
                    let key = key.map_or_else(
                        || Ok(value.clone()),
                        |key| {
                            invoke_callable(program, key.clone(), vec![value.clone()], write_line)
                        },
                    )?;
                    Ok((value, key))
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            keyed.sort_by(|left, right| controlled_value_ordering(&left.1, &right.1));
            if reverse {
                keyed.reverse();
            }
            return Ok(Value::List(Rc::new(RefCell::new(
                keyed.into_iter().map(|(value, _)| value).collect(),
            ))));
        }
        (Value::List(values), "map" | "filter") => {
            let [callable @ (Value::Function(_) | Value::Lambda { .. })] = args.as_slice() else {
                return Err(CompileError::Execution(format!(
                    "{method} expects one function"
                )));
            };
            let mut result = Vec::new();
            for value in values.borrow().iter().cloned() {
                let transformed =
                    invoke_callable(program, callable.clone(), vec![value.clone()], write_line)?;
                if method == "map" {
                    result.push(transformed);
                } else if truthy(&transformed)? {
                    result.push(value);
                }
            }
            return Ok(Value::List(Rc::new(RefCell::new(result))));
        }
        (Value::List(values), "reduce") => {
            let (callable, mut accumulator, start) = match args.as_slice() {
                [callable @ (Value::Function(_) | Value::Lambda { .. })] => {
                    let first = values.borrow().first().cloned().ok_or_else(|| {
                        CompileError::Execution(
                            "reduce of an empty list needs an initial value".into(),
                        )
                    })?;
                    (callable.clone(), first, 1)
                }
                [callable @ (Value::Function(_) | Value::Lambda { .. }), initial] => {
                    (callable.clone(), initial.clone(), 0)
                }
                _ => {
                    return Err(CompileError::Execution(
                        "reduce expects a function and optional initial value".into(),
                    ))
                }
            };
            for value in values.borrow().iter().skip(start) {
                accumulator = invoke_callable(
                    program,
                    callable.clone(),
                    vec![accumulator, value.clone()],
                    write_line,
                )?;
            }
            return Ok(accumulator);
        }
        (Value::List(values), "join") => {
            let [Value::String(separator)] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "join expects one string separator".into(),
                ));
            };
            let parts = values
                .borrow()
                .iter()
                .map(|value| match value {
                    Value::String(value) => Ok(value.clone()),
                    _ => Err(CompileError::Execution(
                        "join requires a list of strings".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Value::String(parts.join(separator)));
        }
        (Value::List(values), "sum") => {
            if !args.is_empty() {
                return Err(CompileError::Execution("sum expects no arguments".into()));
            }
            let values = values.borrow();
            if values.iter().any(|value| matches!(value, Value::Float(_))) {
                let mut total = 0.0;
                for value in values.iter() {
                    total += match value {
                        Value::Int(value) => *value as f64,
                        Value::Float(value) => f64::from_bits(*value),
                        _ => {
                            return Err(CompileError::Execution(
                                "sum requires numeric values".into(),
                            ))
                        }
                    };
                }
                return Ok(Value::Float(total.to_bits()));
            }
            let mut total = 0i64;
            for value in values.iter() {
                let Value::Int(value) = value else {
                    return Err(CompileError::Execution(
                        "sum requires numeric values".into(),
                    ));
                };
                total += value;
            }
            return Ok(Value::Int(total));
        }
        (Value::List(values), "toSet") => {
            if !args.is_empty() {
                return Err(CompileError::Execution("toSet expects no arguments".into()));
            }
            let mut set = Vec::new();
            for value in values.borrow().iter() {
                if !set.contains(value) {
                    set.push(value.clone());
                }
            }
            return Ok(Value::Set(set));
        }
        (Value::List(values), "frequencies") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "frequencies expects no arguments".into(),
                ));
            }
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for value in values.borrow().iter() {
                if let Some((_, Value::Int(count))) =
                    entries.iter_mut().find(|(candidate, _)| candidate == value)
                {
                    *count += 1;
                } else {
                    entries.push((value.clone(), Value::Int(1)));
                }
            }
            return Ok(Value::Map(Rc::new(RefCell::new(entries))));
        }
        (Value::Set(values), "difference") => {
            let [Value::Set(excluded)] = args.as_slice() else {
                return Err(CompileError::Execution("difference expects one set".into()));
            };
            return Ok(Value::Set(
                values
                    .iter()
                    .filter(|value| !excluded.contains(value))
                    .cloned()
                    .collect(),
            ));
        }
        (Value::Set(values), "toList") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "toList expects no arguments".into(),
                ));
            }
            return Ok(Value::List(Rc::new(RefCell::new(values.clone()))));
        }
        (Value::Set(values), "union" | "intersection" | "symmetricDifference") => {
            let [Value::Set(other)] = args.as_slice() else {
                return Err(CompileError::Execution(format!("{method} expects one set")));
            };
            let result = match method {
                "union" => values
                    .iter()
                    .chain(other)
                    .fold(Vec::new(), |mut result, value| {
                        if !result.contains(value) {
                            result.push(value.clone());
                        }
                        result
                    }),
                "intersection" => values
                    .iter()
                    .filter(|value| other.contains(value))
                    .cloned()
                    .collect(),
                _ => values
                    .iter()
                    .filter(|value| !other.contains(value))
                    .chain(other.iter().filter(|value| !values.contains(value)))
                    .cloned()
                    .collect(),
            };
            return Ok(Value::Set(result));
        }
        (Value::List(values), "minimum" | "maximum") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(format!(
                    "{method} expects no arguments"
                )));
            }
            let values = values.borrow();
            let mut best = values.first().cloned().ok_or_else(|| {
                CompileError::Execution(format!("{method} requires a non-empty list"))
            })?;
            for value in values.iter().skip(1) {
                let ordering = controlled_value_ordering(value, &best);
                if (method == "minimum" && ordering.is_lt())
                    || (method == "maximum" && ordering.is_gt())
                {
                    best = value.clone();
                }
            }
            return Ok(best);
        }
        (Value::String(text), "characters") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "characters expects no arguments".into(),
                ));
            }
            return Ok(Value::List(Rc::new(RefCell::new(
                text.chars()
                    .map(|character| Value::String(character.to_string()))
                    .collect(),
            ))));
        }
        (Value::String(text), "words") => {
            if !args.is_empty() {
                return Err(CompileError::Execution("words expects no arguments".into()));
            }
            return Ok(Value::List(Rc::new(RefCell::new(
                text.split_whitespace()
                    .map(|word| Value::String(word.to_owned()))
                    .collect(),
            ))));
        }
        (Value::String(text), "split") => {
            let [Value::String(separator)] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "split expects one string separator".into(),
                ));
            };
            if separator.is_empty() {
                return Err(CompileError::Execution(
                    "split separator cannot be empty".into(),
                ));
            }
            return Ok(Value::List(Rc::new(RefCell::new(
                text.split(separator)
                    .map(|part| Value::String(part.to_owned()))
                    .collect(),
            ))));
        }
        (Value::String(text), "frequencies") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "frequencies expects no arguments".into(),
                ));
            }
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for character in text.chars() {
                let key = Value::String(character.to_string());
                if let Some((_, Value::Int(count))) =
                    entries.iter_mut().find(|(candidate, _)| candidate == &key)
                {
                    *count += 1;
                } else {
                    entries.push((key, Value::Int(1)));
                }
            }
            return Ok(Value::Map(Rc::new(RefCell::new(entries))));
        }
        (Value::String(text), "strip" | "lower" | "upper") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(format!(
                    "{method} expects no arguments"
                )));
            }
            return Ok(Value::String(match method {
                "strip" => text.trim().to_owned(),
                "lower" => text.to_lowercase(),
                _ => text.to_uppercase(),
            }));
        }
        (Value::String(text), "length") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(
                    "length expects no arguments".into(),
                ));
            }
            return Ok(Value::Int(text.chars().count() as i64));
        }
        (Value::String(text), "startsWith" | "endsWith" | "find" | "count") => {
            let [Value::String(needle)] = args.as_slice() else {
                return Err(CompileError::Execution(format!(
                    "{method} expects one string"
                )));
            };
            return Ok(match method {
                "startsWith" => Value::Bool(text.starts_with(needle)),
                "endsWith" => Value::Bool(text.ends_with(needle)),
                "find" => Value::Int(text.find(needle).map_or(-1, |index| index as i64)),
                _ => Value::Int(text.matches(needle).count() as i64),
            });
        }
        (Value::String(text), "replace") => {
            let [Value::String(old), Value::String(new)] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "replace expects two strings".into(),
                ));
            };
            return Ok(Value::String(text.replace(old, new)));
        }
        (Value::Map(entries), "keys" | "values") => {
            if !args.is_empty() {
                return Err(CompileError::Execution(format!(
                    "{method} expects no arguments"
                )));
            }
            let values = entries
                .borrow()
                .iter()
                .map(|(key, value)| if method == "keys" { key } else { value })
                .cloned()
                .collect();
            return Ok(Value::List(Rc::new(RefCell::new(values))));
        }
        (Value::Map(entries), "get") => {
            let [key, fallback] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "get expects a key and fallback".into(),
                ));
            };
            return Ok(entries
                .borrow()
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map_or_else(|| fallback.clone(), |(_, value)| value.clone()));
        }
        (Value::Map(entries), "setDefault") => {
            let [key, fallback] = args.as_slice() else {
                return Err(CompileError::Execution(
                    "setDefault expects a key and fallback".into(),
                ));
            };
            let mut entries = entries.borrow_mut();
            if let Some((_, value)) = entries.iter().find(|(candidate, _)| candidate == key) {
                return Ok(value.clone());
            }
            entries.push((key.clone(), fallback.clone()));
            return Ok(fallback.clone());
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

fn controlled_value_ordering(left: &Value, right: &Value) -> std::cmp::Ordering {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => left.cmp(right),
        (Value::Float(left), Value::Float(right)) => f64::from_bits(*left)
            .partial_cmp(&f64::from_bits(*right))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::Int(left), Value::Float(right)) => (*left as f64)
            .partial_cmp(&f64::from_bits(*right))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::Float(left), Value::Int(right)) => f64::from_bits(*left)
            .partial_cmp(&(*right as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(left), Value::String(right)) => left.cmp(right),
        _ => std::cmp::Ordering::Equal,
    }
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
        (Value::Int(base), BinaryOp::Power, Value::Int(exponent)) if exponent >= 0 => {
            let exponent = u32::try_from(exponent)
                .map_err(|_| CompileError::Execution("integer power overflow".into()))?;
            base.checked_pow(exponent)
                .map(Value::Int)
                .ok_or_else(|| CompileError::Execution("integer power overflow".into()))
        }
        (Value::Int(_), BinaryOp::Power, Value::Int(_)) => Err(CompileError::Execution(
            "negative integer powers require a float base".into(),
        )),
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
        (Value::Float(base), BinaryOp::Power, Value::Float(exponent)) => Ok(float_value(
            f64::from_bits(base).powf(f64::from_bits(exponent)),
        )),
        (Value::Float(base), BinaryOp::Power, Value::Int(exponent)) => {
            Ok(float_value(f64::from_bits(base).powf(exponent as f64)))
        }
        (Value::Int(base), BinaryOp::Power, Value::Float(exponent)) => {
            Ok(float_value((base as f64).powf(f64::from_bits(exponent))))
        }
        (Value::String(left), BinaryOp::Add, Value::String(right)) => {
            Ok(Value::String(left + &right))
        }
        (
            Value::Matrix {
                rows,
                columns,
                fill,
            },
            BinaryOp::Mul,
            Value::Float(factor),
        ) => Ok(Value::Matrix {
            rows,
            columns,
            fill: (f64::from_bits(fill) * f64::from_bits(factor)).to_bits(),
        }),
        (Value::String(needle), BinaryOp::In, Value::String(haystack)) => {
            Ok(Value::Bool(haystack.contains(&needle)))
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

fn comprehension_bindings(
    program: &Program,
    clauses: &[severian_hir::ComprehensionClause],
    variables: &HashMap<String, Value>,
    write_line: &mut dyn FnMut(&str),
) -> Result<Vec<HashMap<String, Value>>, CompileError> {
    fn expand(
        program: &Program,
        clauses: &[severian_hir::ComprehensionClause],
        depth: usize,
        variables: HashMap<String, Value>,
        write_line: &mut dyn FnMut(&str),
        result: &mut Vec<HashMap<String, Value>>,
    ) -> Result<(), CompileError> {
        if depth == clauses.len() {
            result.push(variables);
            return Ok(());
        }
        let clause = &clauses[depth];
        let iterable = evaluate(program, &clause.iterable, &variables, write_line)?;
        for value in iterable_values(iterable)? {
            let mut inner = variables.clone();
            if !match_pattern(program, &value, &clause.pattern, &mut inner) {
                return Err(CompileError::Execution(
                    "comprehension pattern did not match".into(),
                ));
            }
            if let Some(condition) = &clause.condition {
                if !truthy(&evaluate(program, condition, &inner, write_line)?)? {
                    continue;
                }
            }
            expand(program, clauses, depth + 1, inner, write_line, result)?;
        }
        Ok(())
    }

    let mut result = Vec::new();
    expand(
        program,
        clauses,
        0,
        variables.clone(),
        write_line,
        &mut result,
    )?;
    Ok(result)
}

fn range_values(start: i64, end: i64, step: i64) -> Result<Vec<i64>, CompileError> {
    if step == 0 {
        return Err(CompileError::Execution("range step cannot be zero".into()));
    }
    let mut values = Vec::new();
    let mut value = start;
    while if step > 0 { value < end } else { value > end } {
        values.push(value);
        value = value
            .checked_add(step)
            .ok_or_else(|| CompileError::Execution("range overflowed".into()))?;
    }
    Ok(values)
}

fn index_value(object: Value, index: Value) -> Result<Value, CompileError> {
    match (object, index) {
        (Value::List(values), Value::Int(index)) => {
            let values = values.borrow();
            let index = normalize_index(index, values.len(), "list")?;
            values
                .get(index)
                .cloned()
                .ok_or_else(|| CompileError::Execution("list index out of bounds".into()))
        }
        (Value::Tuple(values), Value::Int(index)) => {
            let index = normalize_index(index, values.len(), "tuple")?;
            values
                .get(index)
                .cloned()
                .ok_or_else(|| CompileError::Execution("tuple index out of bounds".into()))
        }
        (Value::String(value), Value::Int(index)) => {
            let length = value.chars().count();
            let index = normalize_index(index, length, "string")?;
            value
                .chars()
                .nth(index)
                .map(|character| Value::String(character.to_string()))
                .ok_or_else(|| CompileError::Execution("string index out of bounds".into()))
        }
        (Value::Map(entries), key) => entries
            .borrow()
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .map(|(_, value)| value.clone())
            .ok_or_else(|| CompileError::Execution("map key not found".into())),
        _ => Err(CompileError::Execution("value is not indexable".into())),
    }
}

fn normalize_index(index: i64, length: usize, kind: &str) -> Result<usize, CompileError> {
    let length = i64::try_from(length)
        .map_err(|_| CompileError::Execution(format!("{kind} is too large to index")))?;
    let index = if index < 0 { length + index } else { index };
    if index < 0 || index >= length {
        return Err(CompileError::Execution(format!(
            "{kind} index out of bounds"
        )));
    }
    Ok(index as usize)
}

fn slice_indices(
    length: usize,
    start: Option<i64>,
    end: Option<i64>,
    step: Option<i64>,
) -> Result<Vec<usize>, CompileError> {
    let length = i64::try_from(length)
        .map_err(|_| CompileError::Execution("collection is too large to slice".into()))?;
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(CompileError::Execution("slice step cannot be zero".into()));
    }
    let mut result = Vec::new();
    if step > 0 {
        let normalize = |value: i64| {
            let value = if value < 0 { value + length } else { value };
            value.clamp(0, length)
        };
        let mut index = start.map(normalize).unwrap_or(0);
        let end = end.map(normalize).unwrap_or(length);
        while index < end {
            result.push(index as usize);
            index += step;
        }
    } else {
        let normalize = |value: i64| {
            let value = if value < 0 { value + length } else { value };
            value.clamp(-1, length - 1)
        };
        let mut index = start.map(normalize).unwrap_or(length - 1);
        let end = end.map(normalize).unwrap_or(-1);
        while index > end {
            result.push(index as usize);
            index += step;
        }
    }
    Ok(result)
}

fn slice_value(
    value: Value,
    start: Option<i64>,
    end: Option<i64>,
    step: Option<i64>,
) -> Result<Value, CompileError> {
    match value {
        Value::List(values) => {
            let values = values.borrow();
            let indices = slice_indices(values.len(), start, end, step)?;
            Ok(Value::List(Rc::new(RefCell::new(
                indices
                    .into_iter()
                    .map(|index| values[index].clone())
                    .collect(),
            ))))
        }
        Value::Tuple(values) => {
            let indices = slice_indices(values.len(), start, end, step)?;
            Ok(Value::Tuple(
                indices
                    .into_iter()
                    .map(|index| values[index].clone())
                    .collect(),
            ))
        }
        Value::String(value) => {
            let characters = value.chars().collect::<Vec<_>>();
            let indices = slice_indices(characters.len(), start, end, step)?;
            Ok(Value::String(
                indices.into_iter().map(|index| characters[index]).collect(),
            ))
        }
        _ => Err(CompileError::Execution("value is not sliceable".into())),
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
        Value::Lambda { .. } => "<lambda>".into(),
        Value::Matrix {
            rows,
            columns,
            fill,
        } => format!("matrix({rows}x{columns}, {})", f64::from_bits(*fill)),
        Value::Tensor { shape, values } => format!(
            "tensor(shape={shape:?}, values=[{}])",
            values
                .iter()
                .map(|value| f64::from_bits(*value).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
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
        Value::Database(_) => "<database>".into(),
        Value::DatabaseServer { address, .. } => format!("<database-server {address}>"),
        Value::DatabaseConnection(_) => "<database-connection>".into(),
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
        Value::ChaosRule {
            function,
            action,
            value,
            ..
        } => {
            let action = match action {
                ChaosAction::Return => "return",
                ChaosAction::Throw => "throw",
            };
            format!("when {function} {action} {}", display_value(value))
        }
        Value::Unit => "unit".into(),
        Value::Break | Value::Continue => "<loop-control>".into(),
    }
}

fn ok_value(value: Value) -> Value {
    Value::Variant {
        name: "ok".into(),
        fields: if value == Value::Unit {
            Vec::new()
        } else {
            vec![value]
        },
    }
}

fn controlled_database_handle(
    value: &Value,
) -> Result<Rc<RefCell<ControlledDatabase>>, CompileError> {
    match value {
        Value::Database(database) | Value::DatabaseConnection(database) => Ok(database.clone()),
        _ => Err(CompileError::Execution(
            "expected a database connection".into(),
        )),
    }
}

fn controlled_database_execute(
    database: &mut ControlledDatabase,
    statement: &str,
) -> Result<i64, CompileError> {
    let normalized = statement.trim().to_ascii_lowercase();
    if normalized.starts_with("create table") {
        return Ok(0);
    }
    if normalized == "begin" || normalized.starts_with("begin ") {
        database.transaction = Some(database.rows.clone());
        return Ok(0);
    }
    if normalized == "commit" {
        database.transaction = None;
        return Ok(0);
    }
    if normalized == "rollback" {
        if let Some(rows) = database.transaction.take() {
            database.rows = rows;
        }
        return Ok(0);
    }
    if normalized.starts_with("delete from") {
        let changed = database.rows.len() as i64;
        database.rows.clear();
        return Ok(changed);
    }
    if normalized.starts_with("insert into") {
        let rows = parse_controlled_insert(statement)?;
        let changed = rows.len() as i64;
        database.rows.extend(rows);
        return Ok(changed);
    }
    Err(CompileError::Execution(format!(
        "controlled database does not support SQL `{statement}`"
    )))
}

fn parse_controlled_insert(statement: &str) -> Result<Vec<Vec<String>>, CompileError> {
    let normalized = statement.to_ascii_lowercase();
    let values_offset = normalized
        .find("values")
        .ok_or_else(|| CompileError::Execution("INSERT requires VALUES".into()))?
        + "values".len();
    let values = &statement[values_offset..];
    let mut groups = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut quoted = false;
    for character in values.chars() {
        match character {
            '\'' => {
                quoted = !quoted;
                if depth > 0 {
                    current.push(character);
                }
            }
            '(' if !quoted => {
                depth += 1;
                if depth > 1 {
                    current.push(character);
                }
            }
            ')' if !quoted => {
                depth -= 1;
                if depth == 0 {
                    groups.push(current.trim().to_owned());
                    current.clear();
                } else {
                    current.push(character);
                }
            }
            _ if depth > 0 => current.push(character),
            _ => {}
        }
    }
    if depth != 0 || quoted || groups.is_empty() {
        return Err(CompileError::Execution("invalid INSERT values".into()));
    }
    Ok(groups
        .into_iter()
        .map(|group| {
            group
                .split(',')
                .map(|cell| cell.trim().trim_matches('\'').to_owned())
                .collect::<Vec<_>>()
        })
        .collect())
}

fn controlled_database_query(database: &ControlledDatabase, statement: &str) -> Value {
    let rows = if statement.to_ascii_lowercase().contains("count(*)") {
        vec![vec![database.rows.len().to_string()]]
    } else {
        database.rows.clone()
    };
    Value::List(Rc::new(RefCell::new(
        rows.into_iter()
            .map(|row| {
                Value::List(Rc::new(RefCell::new(
                    row.into_iter().map(Value::String).collect(),
                )))
            })
            .collect(),
    )))
}

fn decode_json_value(text: &str) -> Result<Value, CompileError> {
    let text = text.trim();
    if text == "true" {
        return Ok(Value::Bool(true));
    }
    if text == "false" {
        return Ok(Value::Bool(false));
    }
    if text == "null" {
        return Ok(Value::Unit);
    }
    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        return Ok(Value::String(
            text[1..text.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\\\", "\\"),
        ));
    }
    if text.starts_with('[') && text.ends_with(']') {
        let body = &text[1..text.len() - 1];
        let values = if body.trim().is_empty() {
            Vec::new()
        } else {
            body.split(',')
                .map(decode_json_value)
                .collect::<Result<Vec<_>, _>>()?
        };
        return Ok(Value::List(Rc::new(RefCell::new(values))));
    }
    if let Ok(value) = text.parse::<i64>() {
        return Ok(Value::Int(value));
    }
    if let Ok(value) = text.parse::<f64>() {
        return Ok(Value::Float(value.to_bits()));
    }
    Err(CompileError::Execution("invalid JSON value".into()))
}

fn encode_json_value(value: &Value) -> String {
    match value {
        Value::Int(value) => value.to_string(),
        Value::Float(value) => f64::from_bits(*value).to_string(),
        Value::Bool(value) => value.to_string(),
        Value::String(value) => format!(
            "\"{}\"",
            value
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
        ),
        Value::List(values) => format!(
            "[{}]",
            values
                .borrow()
                .iter()
                .map(encode_json_value)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Unit => "null".into(),
        _ => format!("\"{}\"", display_value(value)),
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
