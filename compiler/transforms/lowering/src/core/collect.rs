use super::*;

pub(super) fn collect_strings(instructions: &[Instruction], strings: &mut Vec<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => collect_expression_strings(value, strings),
            Instruction::TryLet { value, .. } => {
                strings.push("ok".to_owned());
                collect_expression_strings(value, strings);
            }
            Instruction::Assign { target, value, .. } => {
                collect_expression_strings(target, strings);
                collect_expression_strings(value, strings);
            }
            Instruction::Return(Some(value)) => collect_expression_strings(value, strings),
            Instruction::Return(None) => {}
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_expression_strings(condition, strings);
                collect_strings(then_instructions, strings);
                collect_strings(else_instructions, strings);
            }
            Instruction::While {
                setup,
                condition,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    collect_strings(std::slice::from_ref(setup), strings);
                }
                collect_expression_strings(condition, strings);
                collect_strings(instructions, strings);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                strings.push("ok".to_owned());
                if let Some(setup) = setup {
                    collect_strings(std::slice::from_ref(setup), strings);
                }
                collect_expression_strings(iterable, strings);
                collect_strings(instructions, strings);
            }
            Instruction::Switch { value, arms } => {
                collect_expression_strings(value, strings);
                for arm in arms {
                    collect_pattern_strings(&arm.pattern, strings);
                    if let Some(source) = &arm.source {
                        collect_expression_strings(source, strings);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_strings(guard, strings);
                    }
                    collect_strings(&arm.instructions, strings);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_expression_strings(channel, strings);
                }
                if let Some(setup) = setup {
                    collect_strings(std::slice::from_ref(setup), strings);
                }
                if let Some(condition) = repeat_condition {
                    collect_expression_strings(condition, strings);
                }
                for arm in arms {
                    collect_pattern_strings(&arm.pattern, strings);
                    if let Some(source) = &arm.source {
                        collect_expression_strings(source, strings);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_expression_strings(guard, strings);
                    }
                    collect_strings(&arm.instructions, strings);
                }
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    collect_expression_strings(resource, strings);
                }
                collect_strings(instructions, strings);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

pub(super) fn collect_pattern_strings(pattern: &MatchPattern, strings: &mut Vec<String>) {
    match pattern {
        MatchPattern::String(value) => strings.push(value.clone()),
        MatchPattern::Constructor { name, fields } => {
            strings.push(name.clone());
            for field in fields {
                collect_pattern_strings(field, strings);
            }
        }
        _ => {}
    }
}

pub(super) fn collect_short_circuit_operands<'expression>(
    expression: &'expression Expression,
    op: BinaryOp,
    operands: &mut Vec<&'expression Expression>,
) {
    if let Expression::Binary {
        left,
        op: nested_op,
        right,
    } = expression.kind()
    {
        if *nested_op == op {
            collect_short_circuit_operands(left, op, operands);
            collect_short_circuit_operands(right, op, operands);
            return;
        }
    }
    operands.push(expression);
}

pub(super) fn collect_expression_strings(expression: &Expression, strings: &mut Vec<String>) {
    match expression {
        Expression::Typed { expression, .. } => collect_expression_strings(expression, strings),
        Expression::String(value) => strings.push(value.clone()),
        Expression::Lambda { body, .. } => collect_expression_strings(body, strings),
        Expression::Closure { body, .. } => collect_strings(body, strings),
        Expression::Binary { left, right, .. } => {
            collect_expression_strings(left, strings);
            collect_expression_strings(right, strings);
        }
        Expression::Call { args, .. } | Expression::ForeignCall { args, .. } => {
            for argument in args {
                collect_expression_strings(argument, strings);
            }
        }
        Expression::Format {
            template,
            args,
            arg_types,
        } => {
            strings.push(native_format_template(template, arg_types));
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::List(values) | Expression::Tuple(values) | Expression::Set(values) => {
            for value in values {
                collect_expression_strings(value, strings);
            }
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_expression_strings(key, strings);
                collect_expression_strings(value, strings);
            }
        }
        Expression::Index { object, index } => {
            collect_expression_strings(object, strings);
            collect_expression_strings(index, strings);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_expression_strings(object, strings);
            for bound in [start, end, step].into_iter().flatten() {
                collect_expression_strings(bound, strings);
            }
        }
        Expression::ListComprehension { element, clauses } => {
            collect_expression_strings(element, strings);
            for clause in clauses {
                collect_expression_strings(&clause.iterable, strings);
                if let Some(condition) = &clause.condition {
                    collect_expression_strings(condition, strings);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            collect_expression_strings(element, strings);
            for clause in clauses {
                collect_expression_strings(&clause.iterable, strings);
                if let Some(condition) = &clause.condition {
                    collect_expression_strings(condition, strings);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            collect_expression_strings(key, strings);
            collect_expression_strings(value, strings);
            for clause in clauses {
                collect_expression_strings(&clause.iterable, strings);
                if let Some(condition) = &clause.condition {
                    collect_expression_strings(condition, strings);
                }
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_expression_strings(condition, strings);
            collect_expression_strings(then_expression, strings);
            collect_expression_strings(else_expression, strings);
        }
        Expression::RegistryLookup { key, fallback, .. } => {
            collect_expression_strings(key, strings);
            collect_expression_strings(fallback, strings);
        }
        Expression::Unary { expression, .. } => collect_expression_strings(expression, strings),
        Expression::CallValue { callee, args, .. } => {
            collect_expression_strings(callee, strings);
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::PrintArgs(values) => {
            for value in values {
                collect_expression_strings(value, strings);
            }
        }
        Expression::Construct { class, args, .. } => {
            strings.push(class.clone());
            for value in args {
                collect_expression_strings(value, strings);
            }
        }
        Expression::ConstructFields { class, fields, .. } => {
            strings.push(class.clone());
            for (field, value) in fields {
                strings.push(field.clone());
                collect_expression_strings(value, strings);
            }
        }
        Expression::ObjectUpdate {
            object,
            class,
            fields,
            ..
        } => {
            strings.push(class.clone());
            collect_expression_strings(object, strings);
            for (field, value) in fields {
                strings.push(field.clone());
                collect_expression_strings(value, strings);
            }
        }
        Expression::ObjectDocument { object, fields } => {
            strings.extend(fields.iter().cloned());
            collect_expression_strings(object, strings);
        }
        Expression::Member { object, member } => {
            strings.push(member.clone());
            collect_expression_strings(object, strings);
        }
        Expression::MethodCall {
            object,
            method,
            args,
        } => {
            if matches!(method.as_str(), "filter" | "remove_all") {
                strings.push(String::new());
            }
            collect_expression_strings(object, strings);
            for arg in args {
                collect_expression_strings(arg, strings);
            }
        }
        Expression::Variant { name, fields, .. } => {
            strings.push(name.clone());
            for field in fields {
                collect_expression_strings(field, strings);
            }
        }
        Expression::Task { value, .. } | Expression::Await(value) => {
            collect_expression_strings(value, strings);
        }
        Expression::Channel(capacity) => collect_expression_strings(capacity, strings),
        Expression::Send { value, channel } => {
            collect_expression_strings(value, strings);
            collect_expression_strings(channel, strings);
        }
        Expression::ChaosRule { value, .. } => collect_expression_strings(value, strings),
        _ => {}
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TaskSpec {
    pub(super) function: String,
    pub(super) symbol: String,
    pub(super) params: Vec<ValueType>,
    pub(super) return_type: ValueType,
}

const CHANNEL_MARKER: &str = "<severian-native-channel>";

pub(super) fn task_specs(program: &Program) -> Vec<TaskSpec> {
    let functions = program
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut names = Vec::new();
    for function in &program.functions {
        collect_task_names(&function.instructions, &mut names);
    }
    names.sort();
    names.dedup();
    names
        .into_iter()
        .filter_map(|name| {
            let function = functions.get(name.as_str())?;
            Some(TaskSpec {
                symbol: mangle_symbol_component(&name),
                function: name,
                params: function.params.iter().map(|param| param.ty).collect(),
                return_type: function.return_type,
            })
        })
        .collect()
}

pub(super) fn uses_channels(program: &Program) -> bool {
    let mut names = Vec::new();
    for function in &program.functions {
        collect_task_names(&function.instructions, &mut names);
    }
    names.iter().any(|name| name == CHANNEL_MARKER)
}

pub(super) fn collect_task_names(instructions: &[Instruction], names: &mut Vec<String>) {
    for instruction in instructions {
        match instruction {
            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => collect_task_names_expression(value, names),
            Instruction::Assign { target, value, .. } => {
                collect_task_names_expression(target, names);
                collect_task_names_expression(value, names);
            }
            Instruction::Return(value) => {
                if let Some(value) = value {
                    collect_task_names_expression(value, names);
                }
            }
            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                collect_task_names_expression(condition, names);
                collect_task_names(then_instructions, names);
                collect_task_names(else_instructions, names);
            }
            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    collect_task_names(std::slice::from_ref(setup), names);
                }
                for capability in capabilities {
                    collect_task_names_expression(capability, names);
                }
                collect_task_names_expression(condition, names);
                collect_task_names(instructions, names);
            }
            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    collect_task_names(std::slice::from_ref(setup), names);
                }
                collect_task_names_expression(iterable, names);
                collect_task_names(instructions, names);
            }
            Instruction::Switch { value, arms } => {
                collect_task_names_expression(value, names);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_task_names_expression(source, names);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_task_names_expression(guard, names);
                    }
                    collect_task_names(&arm.instructions, names);
                }
            }
            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                for channel in channels {
                    collect_task_names_expression(channel, names);
                }
                if let Some(setup) = setup {
                    collect_task_names(std::slice::from_ref(setup), names);
                }
                if let Some(condition) = repeat_condition {
                    collect_task_names_expression(condition, names);
                }
                for arm in arms {
                    if let Some(source) = &arm.source {
                        collect_task_names_expression(source, names);
                    }
                    if let Some(guard) = &arm.guard {
                        collect_task_names_expression(guard, names);
                    }
                    collect_task_names(&arm.instructions, names);
                }
            }
            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    collect_task_names_expression(resource, names);
                }
                collect_task_names(instructions, names);
            }
            Instruction::Break | Instruction::Continue => {}
        }
    }
}

pub(super) fn collect_task_names_expression(expression: &Expression, names: &mut Vec<String>) {
    match expression {
        Expression::Typed { expression, .. } => collect_task_names_expression(expression, names),
        Expression::Task { value, .. } => {
            if let Expression::Call { target, .. } = value.kind() {
                names.push(target.name.clone());
            }
            collect_task_names_expression(value, names);
        }
        Expression::Lambda { body, .. } => collect_task_names_expression(body, names),
        Expression::Closure { body, .. } => collect_task_names(body, names),
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                collect_task_names_expression(value, names);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                collect_task_names_expression(value, names);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            collect_task_names_expression(object, names);
            for (_, value) in fields {
                collect_task_names_expression(value, names);
            }
        }
        Expression::ObjectDocument { object, .. } => {
            collect_task_names_expression(object, names);
        }
        Expression::Map(entries) => {
            for (key, value) in entries {
                collect_task_names_expression(key, names);
                collect_task_names_expression(value, names);
            }
        }
        Expression::Index { object, index }
        | Expression::Binary {
            left: object,
            right: index,
            ..
        } => {
            collect_task_names_expression(object, names);
            collect_task_names_expression(index, names);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            collect_task_names_expression(object, names);
            for bound in [start, end, step].into_iter().flatten() {
                collect_task_names_expression(bound, names);
            }
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            collect_task_names_expression(condition, names);
            collect_task_names_expression(then_expression, names);
            collect_task_names_expression(else_expression, names);
        }
        Expression::RegistryLookup { key, fallback, .. } => {
            collect_task_names_expression(key, names);
            collect_task_names_expression(fallback, names);
        }
        Expression::FusedPipeline { input, .. } => collect_task_names_expression(input, names),
        Expression::Ownership { value, .. } => collect_task_names_expression(value, names),
        Expression::Member { object, .. }
        | Expression::Unary {
            expression: object, ..
        }
        | Expression::Await(object)
        | Expression::ChaosRule { value: object, .. } => {
            collect_task_names_expression(object, names)
        }
        Expression::Channel(capacity) => {
            names.push(CHANNEL_MARKER.to_owned());
            collect_task_names_expression(capacity, names);
        }
        Expression::MethodCall { object, args, .. } => {
            collect_task_names_expression(object, names);
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::Send { value, channel } => {
            names.push(CHANNEL_MARKER.to_owned());
            collect_task_names_expression(value, names);
            collect_task_names_expression(channel, names);
        }
        Expression::ListComprehension { element, clauses } => {
            collect_task_names_expression(element, names);
            for clause in clauses {
                collect_task_names_expression(&clause.iterable, names);
                if let Some(condition) = &clause.condition {
                    collect_task_names_expression(condition, names);
                }
            }
        }
        Expression::SetComprehension { element, clauses } => {
            collect_task_names_expression(element, names);
            for clause in clauses {
                collect_task_names_expression(&clause.iterable, names);
                if let Some(condition) = &clause.condition {
                    collect_task_names_expression(condition, names);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            collect_task_names_expression(key, names);
            collect_task_names_expression(value, names);
            for clause in clauses {
                collect_task_names_expression(&clause.iterable, names);
                if let Some(condition) = &clause.condition {
                    collect_task_names_expression(condition, names);
                }
            }
        }
        Expression::Call { args, .. } | Expression::ForeignCall { args, .. } => {
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            collect_task_names_expression(callee, names);
            for arg in args {
                collect_task_names_expression(arg, names);
            }
        }
        Expression::Format { args, .. } => {
            for arg in args {
                collect_task_names_expression(arg, names);
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
