use severian_hir::{Expression, Instruction, Program};
use std::collections::HashSet;

pub(crate) fn fuse_requested_kernels(program: &mut Program) {
    if !program
        .functions
        .iter()
        .any(|function| function.name == "fusedDenseRelu")
    {
        return;
    }

    let mut targets = HashSet::new();
    for function in &program.functions {
        collect_fused_tasks(&function.instructions, &mut targets);
    }
    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            collect_fused_tasks(&function.instructions, &mut targets);
        }
    }

    for function in &mut program.functions {
        if targets.contains(&function.name) {
            rewrite_instructions(&mut function.instructions);
        }
    }
}

fn collect_fused_tasks(instructions: &[Instruction], targets: &mut HashSet<String>) {
    walk_instructions(instructions, &mut |expression| {
        let Expression::Task {
            value, fused: true, ..
        } = expression
        else {
            return;
        };
        if let Expression::Call { function, .. } = value.as_ref() {
            targets.insert(function.clone());
        }
    });
}

fn rewrite_instructions(instructions: &mut [Instruction]) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => rewrite_expression(value),
            Instruction::Assign { target, value, .. } => {
                rewrite_expression(target);
                rewrite_expression(value);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    rewrite_expression(value);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                rewrite_expression(condition);
                rewrite_instructions(then_instructions);
                rewrite_instructions(else_instructions);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    rewrite_instructions(std::slice::from_mut(setup));
                }
                for capability in capabilities {
                    rewrite_expression(capability);
                }
                rewrite_expression(condition);
                rewrite_instructions(instructions);
            }
            Instruction::For {
                iterable,
                instructions,
                ..
            } => {
                rewrite_expression(iterable);
                rewrite_instructions(instructions);
            }
            Instruction::Switch { value, arms } => {
                rewrite_expression(value);
                for arm in arms {
                    if let Some(source) = &mut arm.source {
                        rewrite_expression(source);
                    }
                    if let Some(guard) = &mut arm.guard {
                        rewrite_expression(guard);
                    }
                    rewrite_instructions(&mut arm.instructions);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    rewrite_expression(channel);
                }
                if let Some(setup) = setup {
                    rewrite_instructions(std::slice::from_mut(setup));
                }
                if let Some(condition) = repeat_condition {
                    rewrite_expression(condition);
                }
                for arm in arms {
                    if let Some(source) = &mut arm.source {
                        rewrite_expression(source);
                    }
                    if let Some(guard) = &mut arm.guard {
                        rewrite_expression(guard);
                    }
                    rewrite_instructions(&mut arm.instructions);
                }
            }
            Instruction::With {
                resources,
                instructions,
            } => {
                for resource in resources {
                    rewrite_expression(resource);
                }
                rewrite_instructions(instructions);
            }
        }
    }
}

fn rewrite_expression(expression: &mut Expression) {
    match expression {
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                rewrite_expression(value);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                rewrite_expression(key);
                rewrite_expression(value);
            }
        }
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            rewrite_expression(object);
            rewrite_expression(index);
        }
        Expression::Member { object, .. }
        | Expression::Unary {
            expression: object, ..
        }
        | Expression::Task { value: object, .. }
        | Expression::Await(object)
        | Expression::Channel(object)
        | Expression::ChaosRule { value: object, .. } => rewrite_expression(object),
        Expression::MethodCall { object, args, .. } => {
            rewrite_expression(object);
            for arg in args {
                rewrite_expression(arg);
            }
        }
        Expression::Send { value, channel } => {
            rewrite_expression(value);
            rewrite_expression(channel);
        }
        Expression::ListComprehension {
            element,
            iterable,
            condition,
            ..
        } => {
            rewrite_expression(element);
            rewrite_expression(iterable);
            if let Some(condition) = condition {
                rewrite_expression(condition);
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            rewrite_expression(condition);
            rewrite_expression(then_expression);
            rewrite_expression(else_expression);
        }
        Expression::Call { args, .. } => {
            for arg in args {
                rewrite_expression(arg);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            rewrite_expression(callee);
            for arg in args {
                rewrite_expression(arg);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                rewrite_expression(arg);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }

    let replacement = dense_relu_operands(expression).map(|args| Expression::Call {
        function: "fusedDenseRelu".into(),
        args,
    });
    if let Some(replacement) = replacement {
        *expression = replacement;
    }
}

fn dense_relu_operands(expression: &Expression) -> Option<Vec<Expression>> {
    let Expression::Call {
        function: relu,
        args: relu_args,
    } = expression
    else {
        return None;
    };
    if !matches!(short_name(relu), "activation" | "relu") || relu_args.len() != 1 {
        return None;
    }
    let Expression::Call {
        function: add,
        args: add_args,
    } = &relu_args[0]
    else {
        return None;
    };
    if short_name(add) != "add" || add_args.len() != 2 {
        return None;
    }
    let (matvec, bias) = if is_call(&add_args[0], "matVec") {
        (&add_args[0], &add_args[1])
    } else if is_call(&add_args[1], "matVec") {
        (&add_args[1], &add_args[0])
    } else {
        return None;
    };
    let Expression::Call { args, .. } = matvec else {
        unreachable!()
    };
    if args.len() != 4 {
        return None;
    }
    Some(vec![
        args[0].clone(),
        args[1].clone(),
        args[2].clone(),
        bias.clone(),
        args[3].clone(),
    ])
}

fn is_call(expression: &Expression, expected: &str) -> bool {
    matches!(expression, Expression::Call { function, .. } if short_name(function) == expected)
}

fn short_name(function: &str) -> &str {
    function.rsplit_once('.').map_or(function, |(_, name)| name)
}

fn walk_instructions(instructions: &[Instruction], visit: &mut impl FnMut(&Expression)) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => walk_expression(value, visit),
            Instruction::Assign { target, value, .. } => {
                walk_expression(target, visit);
                walk_expression(value, visit);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    walk_expression(value, visit);
                }
            }
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
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    walk_instructions(std::slice::from_ref(setup), visit);
                }
                for capability in capabilities {
                    walk_expression(capability, visit);
                }
                walk_expression(condition, visit);
                walk_instructions(instructions, visit);
            }
            Instruction::For {
                iterable,
                instructions,
                ..
            } => {
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
            } => {
                for resource in resources {
                    walk_expression(resource, visit);
                }
                walk_instructions(instructions, visit);
            }
        }
    }
}

fn walk_expression(expression: &Expression, visit: &mut impl FnMut(&Expression)) {
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
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            walk_expression(object, visit);
            walk_expression(index, visit);
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
        Expression::ListComprehension {
            element,
            iterable,
            condition,
            ..
        } => {
            walk_expression(element, visit);
            walk_expression(iterable, visit);
            if let Some(condition) = condition {
                walk_expression(condition, visit);
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
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
}
