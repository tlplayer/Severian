use crate::{Diagnostic, DiagnosticBag};
use severian_hir::{Expression, Function, Instruction, Program};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Default)]
pub struct DeadCodeReport {
    pub reachable_functions: BTreeSet<String>,
    pub unreachable_functions: BTreeSet<String>,
    pub never_called_native_symbols: BTreeSet<String>,
}

pub fn analyze(program: &Program) -> DeadCodeReport {
    let mut functions = BTreeMap::<String, &Function>::new();

    for function in &program.functions {
        functions.insert(function.name.clone(), function);
    }

    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            functions.insert(format!("{}::{}", class.name, function.name), function);
        }
    }

    let mut roots = BTreeSet::new();

    if functions.contains_key("main") {
        roots.insert("main".to_owned());
    }

    for (name, function) in &functions {
        if function.native_symbol.is_some() || !function.tests.is_empty() || is_exported(function) {
            roots.insert(name.clone());
        }
    }

    let mut edges = BTreeMap::<String, BTreeSet<String>>::new();
    for (name, function) in &functions {
        let mut calls = BTreeSet::new();
        collect_instruction_calls(&function.instructions, &mut calls);
        for test in &function.tests {
            collect_instruction_calls(&test.instructions, &mut calls);
        }
        edges.insert(name.clone(), calls);
    }

    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from_iter(roots);

    while let Some(function) = queue.pop_front() {
        if !reachable.insert(function.clone()) {
            continue;
        }

        if let Some(calls) = edges.get(&function) {
            for callee in calls {
                if functions.contains_key(callee) && !reachable.contains(callee) {
                    queue.push_back(callee.clone());
                }
            }
        }
    }

    let unreachable_functions = functions
        .keys()
        .filter(|name| !reachable.contains(*name))
        .cloned()
        .collect();

    let never_called_native_symbols = functions
        .iter()
        .filter(|(name, function)| function.native_symbol.is_some() && !reachable.contains(*name))
        .filter_map(|(_, function)| function.native_symbol.clone())
        .collect();

    DeadCodeReport {
        reachable_functions: reachable,
        unreachable_functions,
        never_called_native_symbols,
    }
}

pub fn diagnostics(program: &Program) -> DiagnosticBag {
    let report = analyze(program);
    let mut bag = DiagnosticBag::default();

    for function in report.unreachable_functions {
        bag.push(
            Diagnostic::warning(
                "dead-code::function",
                format!("function `{function}` is unreachable from program roots"),
            )
            .with_help("remove it, export it, test it, or make it reachable from an entry point"),
        );
    }

    bag
}

fn is_exported(function: &Function) -> bool {
    function.decorators.iter().any(|decorator| {
        decorator.package == "export" || decorator.symbols.iter().any(|symbol| symbol == "export")
    })
}

fn collect_instruction_calls(instructions: &[Instruction], calls: &mut BTreeSet<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => collect_expression_calls(value, calls),

            Instruction::Assign { target, value, .. } => {
                collect_expression_calls(target, calls);
                collect_expression_calls(value, calls);
            }

            Instruction::Return(value) => {
                if let Some(value) = value {
                    collect_expression_calls(value, calls);
                }
            }

            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_expression_calls(condition, calls);
                collect_instruction_calls(then_instructions, calls);
                collect_instruction_calls(else_instructions, calls);
            }

            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    collect_instruction_calls(std::slice::from_ref(setup.as_ref()), calls);
                }
                for capability in capabilities {
                    collect_expression_calls(capability, calls);
                }
                collect_expression_calls(condition, calls);
                collect_instruction_calls(instructions, calls);
            }

            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    collect_instruction_calls(std::slice::from_ref(setup.as_ref()), calls);
                }
                collect_expression_calls(iterable, calls);
                collect_instruction_calls(instructions, calls);
            }

            Instruction::Switch { value, arms } => {
                collect_expression_calls(value, calls);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_expression_calls(source, calls);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_calls(guard, calls);
                    }
                    collect_instruction_calls(&arm.instructions, calls);
                }
            }

            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_expression_calls(channel, calls);
                }
                if let Some(setup) = setup {
                    collect_instruction_calls(std::slice::from_ref(setup.as_ref()), calls);
                }
                if let Some(condition) = repeat_condition {
                    collect_expression_calls(condition, calls);
                }
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_expression_calls(source, calls);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_calls(guard, calls);
                    }
                    collect_instruction_calls(&arm.instructions, calls);
                }
            }

            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    collect_expression_calls(resource, calls);
                }
                collect_instruction_calls(instructions, calls);
            }

            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn collect_expression_calls(expression: &Expression, calls: &mut BTreeSet<String>) {
    match expression {
        Expression::Typed { expression, .. } => collect_expression_calls(expression, calls),
        Expression::Call { target, args } => {
            calls.insert(target.name.clone());
            for argument in args {
                collect_expression_calls(argument, calls);
            }
        }

        Expression::Function(function) => {
            calls.insert(function.name.clone());
        }

        Expression::Closure { body, .. } => collect_instruction_calls(body, calls),

        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_expression_calls(value, calls);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                collect_expression_calls(value, calls);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            collect_expression_calls(object, calls);
            for (_, value) in fields {
                collect_expression_calls(value, calls);
            }
        }
        Expression::ObjectDocument { object, .. } => collect_expression_calls(object, calls),

        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_calls(key, calls);
                collect_expression_calls(value, calls);
            }
        }

        Expression::Index { object, index } => {
            collect_expression_calls(object, calls);
            collect_expression_calls(index, calls);
        }

        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_expression_calls(object, calls);
            for bound in [start, end, step].into_iter().flatten() {
                collect_expression_calls(bound, calls);
            }
        }

        Expression::Lambda { body, .. }
        | Expression::Ownership { value: body, .. }
        | Expression::Task { value: body, .. }
        | Expression::Await(body)
        | Expression::Channel(body)
        | Expression::ChaosRule { value: body, .. }
        | Expression::FusedPipeline { input: body, .. }
        | Expression::Unary {
            expression: body, ..
        }
        | Expression::Member { object: body, .. } => collect_expression_calls(body, calls),

        Expression::MethodCall { object, args, .. } => {
            collect_expression_calls(object, calls);
            for argument in args {
                collect_expression_calls(argument, calls);
            }
        }

        Expression::Send { value, channel } => {
            collect_expression_calls(value, calls);
            collect_expression_calls(channel, calls);
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_calls(condition, calls);
            collect_expression_calls(then_expression, calls);
            collect_expression_calls(else_expression, calls);
        }

        Expression::Binary { left, right, .. } => {
            collect_expression_calls(left, calls);
            collect_expression_calls(right, calls);
        }

        Expression::CallValue { callee, args, .. } => {
            collect_expression_calls(callee, calls);
            for argument in args {
                collect_expression_calls(argument, calls);
            }
        }

        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            collect_expression_calls(element, calls);
            for clause in clauses {
                collect_expression_calls(&clause.iterable, calls);
                if let Some(condition) = &clause.condition {
                    collect_expression_calls(condition, calls);
                }
            }
        }

        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            collect_expression_calls(key, calls);
            collect_expression_calls(value, calls);
            for clause in clauses {
                collect_expression_calls(&clause.iterable, calls);
                if let Some(condition) = &clause.condition {
                    collect_expression_calls(condition, calls);
                }
            }
        }

        Expression::Format { args, .. } => {
            for argument in args {
                collect_expression_calls(argument, calls);
            }
        }

        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_) => {}
    }
}
