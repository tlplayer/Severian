use crate::{compile_native, Compilation, CompileError};
use severian_hir::{
    BinaryOp, CallTarget, ContractClause, Expression, Function, FunctionId, FunctionType, HirId,
    Instruction, Program, Test, TestMode, ValueType,
};
use std::collections::HashSet;
use std::path::Path;

/// Build a native executable whose entry point runs every non-integration test.
///
/// This is deliberately unavailable for a test-free library: emitting a no-op
/// entry point would make native acceptance appear to cover code it never ran.
pub fn compile_native_tests(
    compilation: &Compilation,
    output: &Path,
) -> Result<usize, CompileError> {
    compile_selected_native_tests(compilation, output, TestSelection::Unit)
}

pub fn compile_native_profile_tests(
    compilation: &Compilation,
    output: &Path,
) -> Result<usize, CompileError> {
    compile_selected_native_tests(compilation, output, TestSelection::Profile)
}

pub fn compile_native_integration_tests(
    compilation: &Compilation,
    output: &Path,
) -> Result<usize, CompileError> {
    compile_selected_native_tests(compilation, output, TestSelection::Integration)
}

fn compile_selected_native_tests(
    compilation: &Compilation,
    output: &Path,
    selection: TestSelection,
) -> Result<usize, CompileError> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("severian-test-compiler".into())
            .stack_size(16 * 1024 * 1024)
            .spawn_scoped(scope, || {
                let (native, count) = native_test_compilation_selected(compilation, selection)?;
                compile_native(&native, output)?;
                Ok(count)
            })
            .map_err(|error| {
                CompileError::Execution(format!("could not start the test compiler: {error}"))
            })?
            .join()
            .map_err(|_| CompileError::Execution("the test compiler panicked".into()))?
    })
}

pub fn native_test_count(program: &Program) -> usize {
    selected_native_test_count(program, TestSelection::Unit)
}

pub fn native_profile_test_count(program: &Program) -> usize {
    selected_native_test_count(program, TestSelection::Profile)
}

pub fn native_integration_test_count(program: &Program) -> usize {
    selected_native_test_count(program, TestSelection::Integration)
}

fn selected_native_test_count(program: &Program, selection: TestSelection) -> usize {
    program
        .functions
        .iter()
        .flat_map(|function| &function.tests)
        .chain(
            program
                .classes
                .iter()
                .flat_map(|class| class.methods.iter().chain(&class.constructors))
                .flat_map(|function| &function.tests),
        )
        .filter(|test| selected_test(test, selection))
        .count()
}

pub fn native_test_compilation(
    compilation: &Compilation,
) -> Result<(Compilation, usize), CompileError> {
    native_test_compilation_selected(compilation, TestSelection::Unit)
}

pub fn native_profile_test_compilation(
    compilation: &Compilation,
) -> Result<(Compilation, usize), CompileError> {
    native_test_compilation_selected(compilation, TestSelection::Profile)
}

pub fn native_integration_test_compilation(
    compilation: &Compilation,
) -> Result<(Compilation, usize), CompileError> {
    native_test_compilation_selected(compilation, TestSelection::Integration)
}

fn native_test_compilation_selected(
    compilation: &Compilation,
    selection: TestSelection,
) -> Result<(Compilation, usize), CompileError> {
    let mut instructions = Vec::new();
    let mut count = 0;
    for function in &compilation.hir.functions {
        for test in &function.tests {
            if selected_test(test, selection) {
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
                instructions.extend(test_instructions(test));
                count += 1;
            }
        }
    }
    for class in &compilation.hir.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            for test in &function.tests {
                if selected_test(test, selection) {
                    instructions.extend(test_instructions(test));
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
    let mut hir = compilation.optimized_hir.clone();
    const SOURCE_MAIN: &str = "__severian_source_main";
    if let Some(main) = hir
        .functions
        .iter_mut()
        .find(|function| function.name == "main")
    {
        main.name = SOURCE_MAIN.into();
        main.id = FunctionId::from_name(SOURCE_MAIN);
    }
    hir.functions.push(Function {
        // The executable symbol must still be `main`, but its identity must not
        // alias a user-defined main function in source maps or coverage data.
        id: FunctionId::from_name("__severian_test_main"),
        name: "main".into(),
        native_symbol: None,
        decorators: Vec::new(),
        contract: None,
        params: Vec::new(),
        return_type: severian_hir::ValueType::Unit,
        instructions,
        tests: Vec::new(),
    });
    hir.visit_expressions_mut(&mut |expression| {
        if let Expression::Call { target, .. } = expression {
            if target.name == "main" {
                target.name = SOURCE_MAIN.into();
                target.id = FunctionId::from_name(SOURCE_MAIN);
            }
        }
    });
    let mir = severian_mir::lower(&hir);
    let native = Compilation {
        mlir: severian_lowering::lower(&mir),
        optimized_hir: hir.clone(),
        mir,
        hir,
    };
    Ok((native, count))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestSelection {
    Unit,
    Profile,
    Integration,
}

fn selected_test(test: &Test, selection: TestSelection) -> bool {
    match selection {
        TestSelection::Unit => !test.modes.contains(&TestMode::Integration),
        TestSelection::Profile => {
            !test.modes.contains(&TestMode::Integration) && test.modes.contains(&TestMode::Profile)
        }
        TestSelection::Integration => test.modes.contains(&TestMode::Integration),
    }
}

fn test_instructions(test: &Test) -> Vec<Instruction> {
    if !test.modes.contains(&TestMode::Profile) {
        return test.instructions.clone();
    }

    let mut instructions = Vec::new();
    for (name, symbol) in [
        ("__contract_start_time", "__sev_monotonic_ns"),
        ("__contract_start_memory", "__sev_allocation_bytes"),
        ("__contract_start_allocations", "__sev_allocation_count"),
    ] {
        instructions.push(Instruction::Let {
            name: name.into(),
            value: native_metric(symbol),
        });
    }
    instructions.extend(test.instructions.clone());
    for (name, start, symbol) in [
        ("time", "__contract_start_time", "__sev_monotonic_ns"),
        (
            "memory",
            "__contract_start_memory",
            "__sev_allocation_bytes",
        ),
        (
            "allocations",
            "__contract_start_allocations",
            "__sev_allocation_count",
        ),
    ] {
        instructions.push(Instruction::Let {
            name: name.into(),
            value: typed(
                ValueType::Int,
                Expression::Binary {
                    left: Box::new(native_metric(symbol)),
                    op: BinaryOp::Sub,
                    right: Box::new(typed(ValueType::Int, Expression::Variable(start.into()))),
                },
            ),
        });
    }
    instructions.extend(profile_report(test));
    if let Some(contract) = &test.contract {
        instructions.extend(contract.clauses.iter().map(contract_check));
    }
    instructions
}

fn profile_report(test: &Test) -> Vec<Instruction> {
    let name = test.name.as_deref().unwrap_or("unnamed profile test");
    [
        (
            "Profile",
            typed(ValueType::String, Expression::String(name.into())),
        ),
        (
            "time_ns",
            typed(ValueType::Int, Expression::Variable("time".into())),
        ),
        (
            "allocated_bytes",
            typed(ValueType::Int, Expression::Variable("memory".into())),
        ),
        (
            "allocations",
            typed(ValueType::Int, Expression::Variable("allocations".into())),
        ),
    ]
    .into_iter()
    .map(|(label, value)| {
        Instruction::Print(Expression::PrintArgs(vec![
            typed(ValueType::String, Expression::String(label.into())),
            value,
        ]))
    })
    .collect()
}

fn native_metric(symbol: &str) -> Expression {
    typed(
        ValueType::Int,
        Expression::Call {
            target: CallTarget {
                id: FunctionId::from_name(symbol),
                name: symbol.into(),
                native_symbol: Some(symbol.into()),
                signature: Some(FunctionType {
                    parameters: Vec::new(),
                    parameter_any_origins: Vec::new(),
                    returns: ValueType::Int,
                    return_any_origin: None,
                }),
            },
            args: Vec::new(),
        },
    )
}

fn contract_check(clause: &ContractClause) -> Instruction {
    let range = clause
        .condition
        .hir_id()
        .and_then(HirId::legacy_source_range);
    let values = [
        clause
            .message
            .clone()
            .unwrap_or_else(|| "contract condition was not satisfied".into()),
        if clause.location {
            range.map_or_else(
                || "contract source".into(),
                |range| format!("contract source bytes {}..{}", range.start, range.end),
            )
        } else {
            String::new()
        },
        String::new(),
    ];
    let mut arguments = values
        .into_iter()
        .map(|value| typed(ValueType::String, Expression::String(value)))
        .collect::<Vec<_>>();
    if clause.vars && !clause.dependencies.is_empty() {
        arguments[2] = typed(
            ValueType::String,
            Expression::Format {
                template: clause
                    .dependencies
                    .iter()
                    .map(|name| format!("{name}={{}}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                args: clause
                    .dependencies
                    .iter()
                    .zip(&clause.dependency_types)
                    .map(|(name, ty)| typed(*ty, Expression::Variable(name.clone())))
                    .collect(),
                arg_types: clause.dependency_types.clone(),
            },
        );
    }
    let failure = typed(
        ValueType::Unit,
        Expression::Call {
            target: CallTarget {
                id: FunctionId::from_name("__sev_contract_fail"),
                name: "__sev_contract_fail".into(),
                native_symbol: Some("__sev_contract_fail".into()),
                signature: Some(FunctionType {
                    parameters: vec![ValueType::String; 3],
                    parameter_any_origins: vec![None; 3],
                    returns: ValueType::Unit,
                    return_any_origin: None,
                }),
            },
            args: arguments,
        },
    );
    Instruction::If {
        condition: clause.condition.clone(),
        then_instructions: Vec::new(),
        else_instructions: vec![Instruction::Evaluate(failure)],
    }
}

fn typed(ty: ValueType, expression: Expression) -> Expression {
    Expression::Typed {
        id: HirId::synthetic(500),
        ty,
        any_origin: None,
        expression: Box::new(expression),
    }
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
        if let Expression::Call { target, .. } = expression {
            calls.push(target.name.clone());
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
        Expression::Typed { expression, .. } => walk_expression(expression, visit),
        Expression::Closure { body, .. } => walk_instructions(body, visit),
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
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                walk_expression(value, visit);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            walk_expression(object, visit);
            for (_, value) in fields {
                walk_expression(value, visit);
            }
        }
        Expression::ObjectDocument { object, .. } => walk_expression(object, visit),
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
