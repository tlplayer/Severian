use crate::{compile_native, Compilation, CompileError};
use severian_hir::{Expression, Function, Instruction, Program, TestMode};
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
        optimized_hir: hir.clone(),
        hir,
    };
    Ok((native, count))
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
