use severian_hir::{Activation, Expression, Instruction, Program};

pub(crate) fn fuse_activation_chains(program: &mut Program) {
    for global in &mut program.globals {
        rewrite_expression(&mut global.value);
    }
    for function in &mut program.functions {
        rewrite_instructions(&mut function.instructions);
        for test in &mut function.tests {
            rewrite_instructions(&mut test.instructions);
        }
    }
    for class in &mut program.classes {
        for default in class.field_defaults.iter_mut().flatten() {
            rewrite_expression(default);
        }
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            rewrite_instructions(&mut function.instructions);
            for test in &mut function.tests {
                rewrite_instructions(&mut test.instructions);
            }
        }
    }
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
                rewrite_arms(arms);
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
                rewrite_arms(arms);
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

fn rewrite_arms(arms: &mut [severian_hir::SwitchArm]) {
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
        Expression::FusedActivations { input, .. } => rewrite_expression(input),
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

    let replacement = activation_chain(expression);
    if let Some((input, activations)) = replacement {
        *expression = Expression::FusedActivations {
            input: Box::new(input),
            activations,
        };
    }
}

fn activation_chain(expression: &Expression) -> Option<(Expression, Vec<Activation>)> {
    let Expression::Call { function, args } = expression else {
        return None;
    };
    let outer = activation(function)?;
    if args.len() != 1 {
        return None;
    }
    match &args[0] {
        Expression::FusedActivations { input, activations } => {
            if activations.len() >= 16 {
                return None;
            }
            let mut activations = activations.clone();
            activations.push(outer);
            Some((input.as_ref().clone(), activations))
        }
        Expression::Call {
            function: inner,
            args: inner_args,
        } if inner_args.len() == 1 => {
            Some((inner_args[0].clone(), vec![activation(inner)?, outer]))
        }
        _ => None,
    }
}

fn activation(function: &str) -> Option<Activation> {
    match function {
        "models.activation" | "tensor.relu" => Some(Activation::Relu),
        "models.sigmoidActivation" | "tensor.fastSigmoid" => Some(Activation::FastSigmoid),
        "models.tanhActivation" | "tensor.fastTanh" => Some(Activation::FastTanh),
        "models.geluActivation" | "tensor.gelu" => Some(Activation::Gelu),
        "models.swishActivation" | "tensor.swish" => Some(Activation::Swish),
        _ => None,
    }
}
