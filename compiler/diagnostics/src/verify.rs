use crate::{Diagnostic, DiagnosticBag};
use severian_hir::{Expression, Function, Instruction, Program, TensorDimension, ValueType};
use std::collections::BTreeSet;

pub fn verify(program: &Program) -> DiagnosticBag {
    let mut bag = DiagnosticBag::default();

    verify_unique_names(program, &mut bag);

    for class in &program.classes {
        if class.fields.len() != class.field_types.len() {
            bag.push(Diagnostic::error(
                "verify::class-field-types",
                format!(
                    "class `{}` has {} fields but {} field types",
                    class.name,
                    class.fields.len(),
                    class.field_types.len()
                ),
            ));
        }

        if class.fields.len() != class.field_classes.len() {
            bag.push(Diagnostic::error(
                "verify::class-field-classes",
                format!(
                    "class `{}` has {} fields but {} field class entries",
                    class.name,
                    class.fields.len(),
                    class.field_classes.len()
                ),
            ));
        }

        if class.fields.len() != class.field_defaults.len() {
            bag.push(Diagnostic::error(
                "verify::class-field-defaults",
                format!(
                    "class `{}` has {} fields but {} field defaults",
                    class.name,
                    class.fields.len(),
                    class.field_defaults.len()
                ),
            ));
        }

        for function in class.methods.iter().chain(&class.constructors) {
            verify_function(function, &mut bag);
        }
    }

    for function in &program.functions {
        verify_function(function, &mut bag);
    }

    bag
}

fn verify_unique_names(program: &Program, bag: &mut DiagnosticBag) {
    let mut functions = BTreeSet::new();
    let mut function_ids = BTreeSet::new();
    for function in &program.functions {
        if !functions.insert(&function.name) {
            bag.push(Diagnostic::error(
                "verify::duplicate-function",
                format!("duplicate top-level function `{}`", function.name),
            ));
        }
        if !function_ids.insert(function.id) {
            bag.push(Diagnostic::error(
                "verify::duplicate-function-id",
                format!(
                    "function `{}` reuses stable function identity {:?}",
                    function.name, function.id
                ),
            ));
        }
    }

    let mut classes = BTreeSet::new();
    let mut class_ids = BTreeSet::new();
    for class in &program.classes {
        if !classes.insert(&class.name) {
            bag.push(Diagnostic::error(
                "verify::duplicate-class",
                format!("duplicate class `{}`", class.name),
            ));
        }
        if !class_ids.insert(class.id) {
            bag.push(Diagnostic::error(
                "verify::duplicate-class-id",
                format!(
                    "class `{}` reuses stable type identity {:?}",
                    class.name, class.id
                ),
            ));
        }
    }
}

fn verify_function(function: &Function, bag: &mut DiagnosticBag) {
    let mut parameters = BTreeSet::new();
    for parameter in &function.params {
        if !parameters.insert(&parameter.name) {
            bag.push(Diagnostic::error(
                "verify::duplicate-parameter",
                format!(
                    "function `{}` contains duplicate parameter `{}`",
                    function.name, parameter.name
                ),
            ));
        }
        verify_type(parameter.ty, bag);
        if let Some(default) = &parameter.default {
            verify_expression(default, bag);
        }
    }

    verify_type(function.return_type, bag);
    verify_instructions(&function.instructions, bag, 0);

    for test in &function.tests {
        verify_instructions(&test.instructions, bag, 0);
    }
}

fn verify_instructions(instructions: &[Instruction], bag: &mut DiagnosticBag, loop_depth: usize) {
    for instruction in instructions {
        match instruction {
            Instruction::Break | Instruction::Continue if loop_depth == 0 => {
                bag.push(Diagnostic::error(
                    "verify::loop-control",
                    "break/continue appears outside a loop",
                ));
            }

            Instruction::Let { value, .. }
            | Instruction::TryLet { value, .. }
            | Instruction::Print(value)
            | Instruction::Assert(value)
            | Instruction::Evaluate(value) => verify_expression(value, bag),

            Instruction::Assign { target, value, .. } => {
                verify_expression(target, bag);
                verify_expression(value, bag);
            }

            Instruction::Return(value) => {
                if let Some(value) = value {
                    verify_expression(value, bag);
                }
            }

            Instruction::If {
                condition,
                then_instructions,
                else_instructions,
            } => {
                verify_expression(condition, bag);
                verify_instructions(then_instructions, bag, loop_depth);
                verify_instructions(else_instructions, bag, loop_depth);
            }

            Instruction::While {
                setup,
                capabilities,
                condition,
                instructions,
            } => {
                if let Some(setup) = setup {
                    verify_instructions(std::slice::from_ref(setup.as_ref()), bag, loop_depth);
                }
                for capability in capabilities {
                    verify_expression(capability, bag);
                }
                verify_expression(condition, bag);
                verify_instructions(instructions, bag, loop_depth + 1);
            }

            Instruction::For {
                setup,
                iterable,
                instructions,
                ..
            } => {
                if let Some(setup) = setup {
                    verify_instructions(std::slice::from_ref(setup.as_ref()), bag, loop_depth);
                }
                verify_expression(iterable, bag);
                verify_instructions(instructions, bag, loop_depth + 1);
            }

            Instruction::Switch { value, arms } => {
                verify_expression(value, bag);
                for arm in arms {
                    if let Some(source) = &arm.source {
                        verify_expression(source, bag);
                    }
                    if let Some(guard) = &arm.guard {
                        verify_expression(guard, bag);
                    }
                    verify_instructions(&arm.instructions, bag, loop_depth);
                }
            }

            Instruction::ChannelSwitch {
                channels,
                setup,
                repeat_condition,
                arms,
            } => {
                if channels.is_empty() {
                    bag.push(Diagnostic::warning(
                        "verify::empty-channel-switch",
                        "channel switch contains no channels",
                    ));
                }
                for channel in channels {
                    verify_expression(channel, bag);
                }
                if let Some(setup) = setup {
                    verify_instructions(std::slice::from_ref(setup.as_ref()), bag, loop_depth);
                }
                if let Some(condition) = repeat_condition {
                    verify_expression(condition, bag);
                }
                for arm in arms {
                    verify_instructions(&arm.instructions, bag, loop_depth);
                }
            }

            Instruction::With {
                resources,
                instructions,
                ..
            } => {
                for resource in resources {
                    verify_expression(resource, bag);
                }
                verify_instructions(instructions, bag, loop_depth);
            }

            Instruction::Break | Instruction::Continue => {}
        }
    }
}

fn verify_expression(expression: &Expression, bag: &mut DiagnosticBag) {
    match expression {
        Expression::Format {
            args, arg_types, ..
        } if args.len() != arg_types.len() => {
            bag.push(Diagnostic::error(
                "verify::format-arguments",
                format!(
                    "formatted expression has {} arguments but {} argument types",
                    args.len(),
                    arg_types.len()
                ),
            ));
        }

        Expression::CallValue { return_type, .. } => verify_type(*return_type, bag),

        Expression::Closure {
            params,
            body,
            return_type,
        } => {
            for parameter in params {
                verify_type(parameter.ty, bag);
            }
            verify_type(*return_type, bag);
            verify_instructions(body, bag, 0);
        }

        _ => {}
    }

    walk_expression(expression, &mut |nested| {
        if let Expression::CallValue { return_type, .. } = nested {
            verify_type(*return_type, bag);
        }
    });
}

fn verify_type(ty: ValueType, bag: &mut DiagnosticBag) {
    let ValueType::Tensor(tensor) = ty else {
        return;
    };

    if let Some(rank) = tensor.rank {
        if rank > 8 {
            bag.push(Diagnostic::error(
                "verify::tensor-rank",
                format!("tensor rank {rank} exceeds Severian's maximum rank of 8"),
            ));
        }

        for dimension in &tensor.dimensions[rank as usize..] {
            if *dimension != TensorDimension::Dynamic {
                bag.push(Diagnostic::warning(
                    "verify::tensor-tail-dimension",
                    "unused tensor dimension slots should remain Dynamic",
                ));
                break;
            }
        }
    }
}

fn walk_expression(expression: &Expression, visitor: &mut impl FnMut(&Expression)) {
    visitor(expression);
    match expression {
        Expression::Typed { expression, .. } => walk_expression(expression, visitor),
        Expression::Closure { .. } => {}
        Expression::List(values)
        | Expression::Tuple(values)
        | Expression::Set(values)
        | Expression::PrintArgs(values)
        | Expression::Construct { args: values, .. }
        | Expression::Variant { fields: values, .. } => {
            for value in values {
                walk_expression(value, visitor);
            }
        }
        Expression::ConstructFields { fields, .. } => {
            for (_, value) in fields {
                walk_expression(value, visitor);
            }
        }
        Expression::ObjectUpdate { object, fields, .. } => {
            walk_expression(object, visitor);
            for (_, value) in fields {
                walk_expression(value, visitor);
            }
        }
        Expression::ObjectDocument { object, .. } => walk_expression(object, visitor),
        Expression::Map(entries) => {
            for (key, value) in entries {
                walk_expression(key, visitor);
                walk_expression(value, visitor);
            }
        }
        Expression::Index { object, index } => {
            walk_expression(object, visitor);
            walk_expression(index, visitor);
        }
        Expression::Slice {
            object,
            start,
            end,
            step,
        } => {
            walk_expression(object, visitor);
            for bound in [start, end, step].into_iter().flatten() {
                walk_expression(bound, visitor);
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
        | Expression::Member { object: body, .. } => walk_expression(body, visitor),
        Expression::MethodCall { object, args, .. } => {
            walk_expression(object, visitor);
            for argument in args {
                walk_expression(argument, visitor);
            }
        }
        Expression::Send { value, channel } => {
            walk_expression(value, visitor);
            walk_expression(channel, visitor);
        }
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => {
            walk_expression(condition, visitor);
            walk_expression(then_expression, visitor);
            walk_expression(else_expression, visitor);
        }
        Expression::Binary { left, right, .. } => {
            walk_expression(left, visitor);
            walk_expression(right, visitor);
        }
        Expression::Call { args, .. } | Expression::ForeignCall { args, .. } => {
            for argument in args {
                walk_expression(argument, visitor);
            }
        }
        Expression::CallValue { callee, args, .. } => {
            walk_expression(callee, visitor);
            for argument in args {
                walk_expression(argument, visitor);
            }
        }
        Expression::ListComprehension { element, clauses }
        | Expression::SetComprehension { element, clauses } => {
            walk_expression(element, visitor);
            for clause in clauses {
                walk_expression(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    walk_expression(condition, visitor);
                }
            }
        }
        Expression::MapComprehension {
            key,
            value,
            clauses,
        } => {
            walk_expression(key, visitor);
            walk_expression(value, visitor);
            for clause in clauses {
                walk_expression(&clause.iterable, visitor);
                if let Some(condition) = &clause.condition {
                    walk_expression(condition, visitor);
                }
            }
        }
        Expression::Format { args, .. } => {
            for argument in args {
                walk_expression(argument, visitor);
            }
        }
        Expression::RegistryLookup { key, fallback, .. } => {
            walk_expression(key, visitor);
            walk_expression(fallback, visitor);
        }
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Boolean(_)
        | Expression::String(_)
        | Expression::Variable(_)
        | Expression::Function(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::FunctionId;

    fn function(id: FunctionId, name: &str) -> Function {
        Function {
            id,
            name: name.into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: Vec::new(),
            return_type: ValueType::Unit,
            instructions: Vec::new(),
            tests: Vec::new(),
        }
    }

    #[test]
    fn rejects_distinct_names_that_reuse_a_stable_function_id() {
        let id = FunctionId::from_name("first");
        let program = Program {
            metadata: Default::default(),
            globals: Vec::new(),
            classes: Vec::new(),
            functions: vec![function(id, "first"), function(id, "renamed")],
        };

        let diagnostics = verify(&program);
        assert!(diagnostics
            .diagnostics()
            .iter()
            .any(|diagnostic| { diagnostic.code.0 == "verify::duplicate-function-id" }));
    }
}
