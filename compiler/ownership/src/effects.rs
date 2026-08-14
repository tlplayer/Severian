use super::*;

pub(super) fn effect_name(effect: ParameterEffect) -> &'static str {
    match effect {
        ParameterEffect::View => "shared view",
        ParameterEffect::Borrow => "exclusive borrow",
        ParameterEffect::Move => "ownership transfer",
    }
}

pub(super) fn infer_function_effects(
    program: &Program,
) -> HashMap<FunctionId, Vec<ParameterEffect>> {
    let mut effects = HashMap::new();
    for function in &program.functions {
        effects.insert(function.id, infer_parameter_effects(function));
    }
    for class in &program.classes {
        for function in class.methods.iter().chain(&class.constructors) {
            effects
                .entry(function.id)
                .or_insert_with(|| infer_parameter_effects(function));
        }
    }
    effects
}

pub(super) fn infer_parameter_effects(function: &Function) -> Vec<ParameterEffect> {
    let parameters = function
        .params
        .iter()
        .enumerate()
        .map(|(index, parameter)| (parameter.name.id, index))
        .collect::<HashMap<_, _>>();
    let mut effects = vec![ParameterEffect::View; function.params.len()];
    infer_instruction_effects(&function.instructions, &parameters, &mut effects);
    effects
}

pub(super) fn infer_instruction_effects(
    instructions: &[Instruction],
    parameters: &HashMap<BindingId, usize>,
    effects: &mut [ParameterEffect],
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => {
                infer_expression_effect(value, Access::Read, parameters, effects)
            }
            Instruction::Assign { target, value, .. } => {
                infer_expression_effect(value, Access::Read, parameters, effects);
                infer_expression_effect(target, Access::Mutate, parameters, effects);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    infer_expression_effect(value, Access::Read, parameters, effects);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                infer_expression_effect(condition, Access::Read, parameters, effects);
                infer_instruction_effects(then_instructions, parameters, effects);
                infer_instruction_effects(else_instructions, parameters, effects);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                for capability in capabilities {
                    infer_expression_effect(capability, Access::Read, parameters, effects);
                }
                infer_expression_effect(condition, Access::Read, parameters, effects);
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                infer_expression_effect(iterable, Access::Read, parameters, effects);
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::Switch { value, arms } => {
                infer_expression_effect(value, Access::Read, parameters, effects);
                infer_arm_effects(arms, parameters, effects);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    infer_expression_effect(channel, Access::Read, parameters, effects);
                }
                if let Some(setup) = setup {
                    infer_instruction_effects(std::slice::from_ref(setup), parameters, effects);
                }
                if let Some(condition) = repeat_condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
                infer_arm_effects(arms, parameters, effects);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    infer_expression_effect(resource, Access::Read, parameters, effects);
                }
                infer_instruction_effects(instructions, parameters, effects);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

pub(super) fn infer_arm_effects(
    arms: &[SwitchArm],
    parameters: &HashMap<BindingId, usize>,
    effects: &mut [ParameterEffect],
) {
    for arm in arms {
        if let Some(source) = &arm.source {
            infer_expression_effect(source, Access::Read, parameters, effects);
        }
        if let Some(guard) = &arm.guard {
            infer_expression_effect(guard, Access::Read, parameters, effects);
        }
        infer_instruction_effects(&arm.instructions, parameters, effects);
    }
}

pub(super) fn infer_expression_effect(
    expression: &Expression,
    access: Access,
    parameters: &HashMap<BindingId, usize>,
    effects: &mut [ParameterEffect],
) {
    match expression {
        Expression::Typed { expression, .. } => {
            infer_expression_effect(expression, access, parameters, effects)
        }
        Expression::Variable(name) => mark_parameter_effect(
            name,
            if access == Access::Mutate {
                ParameterEffect::Borrow
            } else {
                ParameterEffect::View
            },
            parameters,
            effects,
        ),
        Expression::Ownership { op, value } => {
            let effect = match op {
                OwnershipOp::Move => ParameterEffect::Move,
                OwnershipOp::Borrow => ParameterEffect::Borrow,
                OwnershipOp::View | OwnershipOp::Clone | OwnershipOp::AddressOf => {
                    ParameterEffect::View
                }
            };
            if let Expression::Variable(name) = value.kind() {
                mark_parameter_effect(name, effect, parameters, effects);
            } else {
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::Member { object, .. }
        | Expression::Await(object)
        | Expression::Channel(object)
        | Expression::Task { value: object, .. }
        | Expression::ChaosRule { value: object, .. }
        | Expression::FusedPipeline { input: object, .. }
        | Expression::Unary {
            expression: object, ..
        } => infer_expression_effect(object, access, parameters, effects),
        Expression::Lambda { body, .. } => {
            infer_expression_effect(body, Access::Read, parameters, effects)
        }
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            infer_expression_effect(object, Access::Read, parameters, effects);
            for (_, value) in fields {
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::ObjectDocument { object, .. } => {
            infer_expression_effect(object, Access::Read, parameters, effects);
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                infer_expression_effect(key, Access::Read, parameters, effects);
                infer_expression_effect(value, Access::Read, parameters, effects);
            }
        }
        Expression::Index { object, index } => {
            infer_expression_effect(object, access, parameters, effects);
            infer_expression_effect(index, Access::Read, parameters, effects);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            infer_expression_effect(object, access, parameters, effects);
            for bound in [start, end, step].into_iter().flatten() {
                infer_expression_effect(bound, Access::Read, parameters, effects);
            }
        }
        Expression::MethodCall {
            object,
            method,
            args,
        } => {
            let receiver_access = if mutating_method(method) {
                Access::Mutate
            } else {
                Access::Read
            };
            infer_expression_effect(object, receiver_access, parameters, effects);
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::Send { value, channel } => {
            infer_expression_effect(value, Access::Read, parameters, effects);
            infer_expression_effect(channel, Access::Mutate, parameters, effects);
        }
        Expression::ListComprehension { element, clauses } => {
            infer_expression_effect(element, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            infer_expression_effect(element, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            infer_expression_effect(key, Access::Read, parameters, effects);
            infer_expression_effect(value, Access::Read, parameters, effects);
            for clause in clauses {
                infer_expression_effect(&clause.iterable, Access::Read, parameters, effects);
                if let Some(condition) = &clause.condition {
                    infer_expression_effect(condition, Access::Read, parameters, effects);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            infer_expression_effect(condition, Access::Read, parameters, effects);
            infer_expression_effect(then_expression, access, parameters, effects);
            infer_expression_effect(else_expression, access, parameters, effects);
        }
        Expression::Binary { left, right, .. } => {
            infer_expression_effect(left, Access::Read, parameters, effects);
            infer_expression_effect(right, Access::Read, parameters, effects);
        }
        Expression::Format { args, .. } | Expression::Call { args, .. } => {
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            infer_expression_effect(callee, Access::Read, parameters, effects);
            for arg in args {
                infer_expression_effect(arg, Access::Read, parameters, effects);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

pub(super) fn mark_parameter_effect(
    binding: &BindingRef,
    effect: ParameterEffect,
    parameters: &HashMap<BindingId, usize>,
    effects: &mut [ParameterEffect],
) {
    if let Some(index) = parameters.get(&binding.id) {
        effects[*index] = effects[*index].max(effect);
    }
}

pub(super) fn define_pattern(checker: &mut Checker, pattern: &MatchPattern) {
    match pattern {
        MatchPattern::Bind(name) => checker.define(name.clone(), None),
        MatchPattern::Constructor { fields, .. } => {
            for field in fields {
                define_pattern(checker, field);
            }
        }
        _ => {}
    }
}

pub(super) fn mutating_method(method: &str) -> bool {
    matches!(
        method,
        "append"
            | "append_left"
            | "appendleft"
            | "extend"
            | "push"
            | "pop"
            | "pop_left"
            | "popleft"
            | "remove"
            | "clear"
            | "insert"
            | "sort"
            | "reverse"
            | "heapPush"
            | "heapPop"
            | "heap_push"
            | "heap_pop"
            | "setDefault"
            | "set_default"
            | "set"
    )
}

pub(super) fn count_instruction_uses(instructions: &[Instruction]) -> HashMap<BindingId, usize> {
    let mut counts = HashMap::new();
    count_instructions(instructions, &mut counts);
    counts
}

pub(super) fn count_instructions(
    instructions: &[Instruction],
    counts: &mut HashMap<BindingId, usize>,
) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => count_expression(value, counts),
            Instruction::Assign { target, value, .. } => {
                count_expression(target, counts);
                count_expression(value, counts);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    count_expression(value, counts);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                count_expression(condition, counts);
                count_instructions(then_instructions, counts);
                count_instructions(else_instructions, counts);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                for capability in capabilities {
                    count_expression(capability, counts);
                }
                count_expression(condition, counts);
                count_instructions(instructions, counts);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                count_expression(iterable, counts);
                count_instructions(instructions, counts);
            }
            Instruction::Switch { value, arms } => {
                count_expression(value, counts);
                count_arms(arms, counts);
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    count_expression(channel, counts);
                }
                if let Some(setup) = setup {
                    count_instructions(std::slice::from_ref(setup), counts);
                }
                if let Some(condition) = repeat_condition {
                    count_expression(condition, counts);
                }
                count_arms(arms, counts);
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    count_expression(resource, counts);
                }
                count_instructions(instructions, counts);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

pub(super) fn count_arms(arms: &[SwitchArm], counts: &mut HashMap<BindingId, usize>) {
    for arm in arms {
        if let Some(source) = &arm.source {
            count_expression(source, counts);
        }
        if let Some(guard) = &arm.guard {
            count_expression(guard, counts);
        }
        count_instructions(&arm.instructions, counts);
    }
}

pub(super) fn count_expression(expression: &Expression, counts: &mut HashMap<BindingId, usize>) {
    match expression {
        Expression::Typed { expression, .. } => count_expression(expression, counts),
        Expression::Variable(binding) => *counts.entry(binding.id).or_default() += 1,
        Expression::Ownership { value, .. }
        | Expression::Member { object: value, .. }
        | Expression::Await(value)
        | Expression::Channel(value)
        | Expression::Task { value, .. }
        | Expression::ChaosRule { value, .. }
        | Expression::FusedPipeline { input: value, .. }
        | Expression::Unary {
            expression: value, ..
        } => count_expression(value, counts),
        Expression::Lambda { body, .. } => count_expression(body, counts),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                count_expression(value, counts);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                count_expression(value, counts);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            count_expression(object, counts);
            for (_, value) in fields {
                count_expression(value, counts);
            }
        }
        Expression::ObjectDocument { object, .. } => count_expression(object, counts),
        Expression::Map(entries) => {
            for (key, value) in entries {
                count_expression(key, counts);
                count_expression(value, counts);
            }
        }
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            count_expression(object, counts);
            count_expression(index, counts);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            count_expression(object, counts);
            for bound in [start, end, step].into_iter().flatten() {
                count_expression(bound, counts);
            }
        }
        Expression::Format { args, .. } | Expression::Call { args, .. } => {
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::MethodCall { object, args, .. } => {
            count_expression(object, counts);
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::Send { value, channel } => {
            count_expression(value, counts);
            count_expression(channel, counts);
        }
        Expression::ListComprehension { element, clauses } => {
            count_expression(element, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            count_expression(element, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            count_expression(key, counts);
            count_expression(value, counts);
            for clause in clauses {
                count_expression(&clause.iterable, counts);
                if let Some(condition) = &clause.condition {
                    count_expression(condition, counts);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            count_expression(condition, counts);
            count_expression(then_expression, counts);
            count_expression(else_expression, counts);
        }
        Expression::CallValue { callee, args, .. } => {
            count_expression(callee, counts);
            for arg in args {
                count_expression(arg, counts);
            }
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Function(_) => {}
    }
}

pub(super) fn ownership_error(code: &str, message: String) -> OwnershipError {
    OwnershipError {
        message: format!("{code}: {message}"),
    }
}

pub(super) fn unknown(binding: &BindingRef) -> OwnershipError {
    ownership_error(
        "E0300",
        format!("ownership operation references unknown binding `{binding}`"),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnershipError {
    pub message: String,
}

impl std::fmt::Display for OwnershipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for OwnershipError {}
