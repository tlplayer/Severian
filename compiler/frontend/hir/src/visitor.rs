use crate::*;

pub(crate) fn visit_function_expressions_mut(
    function: &mut Function,
    visitor: &mut impl FnMut(&mut Expression),
) {
    for parameter in &mut function.params {
        if let Some(default) = &mut parameter.default {
            visit_expression_mut(default, visitor);
        }
    }
    if let Some(contract) = &mut function.contract {
        for clause in &mut contract.clauses {
            visit_expression_mut(&mut clause.condition, visitor);
        }
        for capability in &mut contract.capabilities {
            visit_expression_mut(capability, visitor);
        }
    }
    visit_instructions_mut(&mut function.instructions, visitor);
    for test in &mut function.tests {
        if let Some(contract) = &mut test.contract {
            for clause in &mut contract.clauses {
                visit_expression_mut(&mut clause.condition, visitor);
            }
            for capability in &mut contract.capabilities {
                visit_expression_mut(capability, visitor);
            }
        }
        visit_instructions_mut(&mut test.instructions, visitor);
    }
}

pub(crate) fn visit_instructions_mut(
    instructions: &mut [Instruction],
    visitor: &mut impl FnMut(&mut Expression),
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => visit_expression_mut(value, visitor),
            Instruction::Assign { target, value, .. } => {
                visit_expression_mut(target, visitor);
                visit_expression_mut(value, visitor);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    visit_expression_mut(value, visitor);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                visit_expression_mut(condition, visitor);
                visit_instructions_mut(then_instructions, visitor);
                visit_instructions_mut(else_instructions, visitor);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                for capability in capabilities {
                    visit_expression_mut(capability, visitor);
                }
                visit_expression_mut(condition, visitor);
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                visit_expression_mut(iterable, visitor);
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::Switch { value, arms } => {
                visit_expression_mut(value, visitor);
                visit_arms_mut(arms, visitor);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    visit_expression_mut(channel, visitor);
                }
                if let Some(setup) = setup {
                    visit_instructions_mut(std::slice::from_mut(setup), visitor);
                }
                if let Some(condition) = repeat_condition {
                    visit_expression_mut(condition, visitor);
                }
                visit_arms_mut(arms, visitor);
            }
            Instruction::With {
                placement: _,
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    visit_expression_mut(resource, visitor);
                }
                visit_instructions_mut(instructions, visitor);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

pub(crate) fn visit_arms_mut(arms: &mut [SwitchArm], visitor: &mut impl FnMut(&mut Expression)) {
    for arm in arms {
        if let Some(source) = &mut arm.source {
            visit_expression_mut(source, visitor);
        }
        if let Some(guard) = &mut arm.guard {
            visit_expression_mut(guard, visitor);
        }
        visit_instructions_mut(&mut arm.instructions, visitor);
    }
}

pub(crate) fn visit_expression_mut(
    expression: &mut Expression,
    visitor: &mut impl FnMut(&mut Expression),
) {
    match expression {
        Expression::Typed { expression, .. } => visit_expression_mut(expression, visitor),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                visit_expression_mut(value, visitor);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                visit_expression_mut(value, visitor);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            visit_expression_mut(object, visitor);
            for (_, value) in fields {
                visit_expression_mut(value, visitor);
            }
        }
        Expression::ObjectDocument { object, .. } => visit_expression_mut(object, visitor),
        Expression::Map(entries) => {
            for (key, value) in entries {
                visit_expression_mut(key, visitor);
                visit_expression_mut(value, visitor);
            }
        }
        Expression::Index { object, index } => {
            visit_expression_mut(object, visitor);
            visit_expression_mut(index, visitor);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            visit_expression_mut(object, visitor);
            for bound in [start, end, step].into_iter().flatten() {
                visit_expression_mut(bound, visitor);
            }
        }
        Expression::Member { object, .. }
        | Expression::Await(object)
        | Expression::Channel(object)
        | Expression::ChaosRule { value: object, .. }
        | Expression::FusedPipeline { input: object, .. } => {
            visit_expression_mut(object, visitor);
        }
        Expression::Ownership { value, .. } => visit_expression_mut(value, visitor),
        Expression::Lambda { body, .. } => visit_expression_mut(body, visitor),
        Expression::Closure { body, .. } => visit_instructions_mut(body, visitor),
        Expression::MethodCall { object, args, .. } => {
            visit_expression_mut(object, visitor);
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Task { value, .. } => visit_expression_mut(value, visitor),
        Expression::Send { value, channel } => {
            visit_expression_mut(value, visitor);
            visit_expression_mut(channel, visitor);
        }
        Expression::ListComprehension { element, clauses } => {
            visit_expression_mut(element, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            visit_expression_mut(element, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            visit_expression_mut(key, visitor);
            visit_expression_mut(value, visitor);
            for clause in clauses {
                visit_expression_mut(&mut clause.iterable, visitor);
                if let Some(condition) = &mut clause.condition {
                    visit_expression_mut(condition, visitor);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            visit_expression_mut(condition, visitor);
            visit_expression_mut(then_expression, visitor);
            visit_expression_mut(else_expression, visitor);
        }
        Expression::RegistryLookup { key, fallback, .. } => {
            visit_expression_mut(key, visitor);
            visit_expression_mut(fallback, visitor);
        }
        Expression::Unary { expression, .. } => visit_expression_mut(expression, visitor),
        Expression::Binary { left, right, .. } => {
            visit_expression_mut(left, visitor);
            visit_expression_mut(right, visitor);
        }
        Expression::Call { args, .. } | Expression::ForeignCall { args, .. } => {
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            visit_expression_mut(callee, visitor);
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                visit_expression_mut(arg, visitor);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
    visitor(expression);
}
