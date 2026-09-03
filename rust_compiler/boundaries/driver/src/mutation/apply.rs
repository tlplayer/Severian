use super::model::{Mutation, MutationEdit};
use severian_ast::{
    ComprehensionClause, Expression, ExpressionKind, FunctionContract, FunctionDeclaration,
    GenericConstraint, Item, Literal, PropertyDeclaration, Statement, UnaryOperator,
};
use severian_modules::ModuleGraph;

pub(crate) fn apply(graph: &mut ModuleGraph, mutation: &Mutation) -> bool {
    let Some(module) = graph
        .modules
        .iter_mut()
        .find(|module| module.path == mutation.file)
    else {
        return false;
    };
    module
        .ast
        .items
        .iter_mut()
        .any(|item| apply_item(item, mutation.edit))
}

fn apply_item(item: &mut Item, edit: MutationEdit) -> bool {
    match item {
        Item::Import(_) | Item::Test(_) => false,
        Item::Binding(binding) => apply_expression(&mut binding.value, edit),
        Item::Expression(expression) => apply_expression(expression, edit),
        Item::Function(function) => apply_function(function, edit),
        Item::Type(declaration) => apply_constraints(&mut declaration.constraints, edit),
        Item::Trait(declaration) => {
            apply_constraints(&mut declaration.constraints, edit)
                || declaration
                    .properties
                    .iter_mut()
                    .any(|property| apply_property(property, edit))
                || declaration
                    .methods
                    .iter_mut()
                    .any(|function| apply_function(function, edit))
        }
        Item::Class(declaration) => {
            apply_constraints(&mut declaration.constraints, edit)
                || declaration
                    .fields
                    .iter_mut()
                    .any(|field| apply_property(field, edit))
                || declaration
                    .constructors
                    .iter_mut()
                    .chain(&mut declaration.methods)
                    .any(|function| apply_function(function, edit))
                || declaration.operators.iter_mut().any(|operator| {
                    apply_contracts(&mut operator.contracts, edit)
                        || apply_statements(&mut operator.body, edit)
                })
        }
        Item::Extension(declaration) => {
            declaration
                .methods
                .iter_mut()
                .any(|function| apply_function(function, edit))
                || declaration.operators.iter_mut().any(|operator| {
                    apply_contracts(&mut operator.contracts, edit)
                        || apply_statements(&mut operator.body, edit)
                })
        }
        Item::Enum(declaration) => declaration.variants.iter_mut().any(|variant| {
            variant
                .fields
                .iter_mut()
                .any(|field| apply_property(field, edit))
        }),
    }
}

fn apply_function(function: &mut FunctionDeclaration, edit: MutationEdit) -> bool {
    apply_constraints(&mut function.constraints, edit)
        || apply_contracts(&mut function.contracts, edit)
        || function.parameters.iter_mut().any(|parameter| {
            parameter
                .default
                .as_mut()
                .is_some_and(|value| apply_expression(value, edit))
        })
        || function.hook.as_mut().is_some_and(|hook| {
            apply_statements(&mut hook.with_phase, edit)
                || apply_statements(&mut hook.without_phase, edit)
        })
        || function
            .body
            .as_mut()
            .is_some_and(|body| apply_statements(body, edit))
}

fn apply_property(property: &mut PropertyDeclaration, edit: MutationEdit) -> bool {
    property
        .default
        .as_mut()
        .is_some_and(|value| apply_expression(value, edit))
        || property.constraints.iter_mut().any(|constraint| {
            apply_expression(&mut constraint.condition, edit)
                || constraint
                    .failure
                    .as_mut()
                    .is_some_and(|failure| apply_expression(failure, edit))
        })
}

fn apply_constraints(constraints: &mut [GenericConstraint], edit: MutationEdit) -> bool {
    constraints.iter_mut().any(|constraint| match constraint {
        GenericConstraint::Predicate(expression) => apply_expression(expression, edit),
        GenericConstraint::Parameter { .. } | GenericConstraint::VariadicPack { .. } => false,
    })
}

fn apply_contracts(contracts: &mut [FunctionContract], edit: MutationEdit) -> bool {
    contracts.iter_mut().any(|contract| {
        apply_expression(&mut contract.condition, edit)
            || contract
                .failure
                .as_mut()
                .is_some_and(|failure| apply_expression(failure, edit))
    })
}

fn apply_statements(statements: &mut [Statement], edit: MutationEdit) -> bool {
    statements
        .iter_mut()
        .any(|statement| apply_statement(statement, edit))
}

fn apply_statement(statement: &mut Statement, edit: MutationEdit) -> bool {
    match statement {
        Statement::Binding(binding) => apply_expression(&mut binding.value, edit),
        Statement::Destructure { value, .. } => apply_expression(value, edit),
        Statement::FieldAssignment { object, value, .. } => {
            apply_expression(object, edit) || apply_expression(value, edit)
        }
        Statement::IndexAssignment {
            object,
            index,
            value,
            ..
        } => {
            apply_expression(object, edit)
                || apply_expression(index, edit)
                || apply_expression(value, edit)
        }
        Statement::Expression(expression) | Statement::Defer { expression, .. } => {
            apply_expression(expression, edit)
        }
        Statement::Return { value, .. } => value
            .as_mut()
            .is_some_and(|value| apply_expression(value, edit)),
        Statement::Assert {
            condition, message, ..
        } => {
            apply_expression(condition, edit)
                || message
                    .as_mut()
                    .is_some_and(|message| apply_expression(message, edit))
        }
        Statement::Unsafe { body, .. } | Statement::Placement { body, .. } => {
            apply_statements(body, edit)
        }
        Statement::Try {
            body, catch_body, ..
        } => apply_statements(body, edit) || apply_statements(catch_body, edit),
        Statement::FallibleElse { value, body, .. } => {
            apply_expression(value, edit) || apply_statements(body, edit)
        }
        Statement::If {
            condition,
            then_block,
            else_block,
            ..
        } => {
            if matches!(edit, MutationEdit::NegateConditional { condition: span } if span == condition.span)
            {
                let operand = condition.clone();
                condition.kind = ExpressionKind::Unary {
                    operator: UnaryOperator::Not,
                    operand: Box::new(operand),
                };
                true
            } else {
                apply_expression(condition, edit)
                    || apply_statements(then_block, edit)
                    || apply_statements(else_block, edit)
            }
        }
        Statement::While {
            condition,
            initializer,
            guards,
            body,
            ..
        } => {
            apply_expression(condition, edit)
                || initializer
                    .as_mut()
                    .is_some_and(|binding| apply_expression(&mut binding.value, edit))
                || guards
                    .iter_mut()
                    .any(|guard| apply_expression(&mut guard.condition, edit))
                || apply_statements(body, edit)
        }
        Statement::For {
            iterable,
            initializer,
            body,
            ..
        } => {
            apply_expression(iterable, edit)
                || initializer
                    .as_mut()
                    .is_some_and(|binding| apply_expression(&mut binding.value, edit))
                || apply_statements(body, edit)
        }
        Statement::Match { subject, cases, .. } => {
            apply_expression(subject, edit)
                || cases
                    .iter_mut()
                    .any(|case| apply_statements(&mut case.body, edit))
        }
        Statement::Select {
            limit,
            cases,
            error_body,
            ..
        } => {
            apply_expression(limit, edit)
                || cases.iter_mut().any(|case| {
                    apply_expression(&mut case.channel, edit)
                        || apply_statements(&mut case.body, edit)
                })
                || apply_statements(error_body, edit)
        }
        Statement::Break { .. } | Statement::Continue { .. } => false,
    }
}

fn apply_expression(expression: &mut Expression, edit: MutationEdit) -> bool {
    match edit {
        MutationEdit::Boolean {
            expression: span,
            value,
        } if expression.span == span
            && matches!(&expression.kind, ExpressionKind::Literal(Literal::Boolean(current)) if *current == value) =>
        {
            expression.kind = ExpressionKind::Literal(Literal::Boolean(!value));
            return true;
        }
        MutationEdit::Binary {
            expression: span,
            original,
            replacement,
        } if expression.span == span
            && matches!(&expression.kind, ExpressionKind::Binary { operator, .. } if *operator == original) =>
        {
            let ExpressionKind::Binary { operator, .. } = &mut expression.kind else {
                unreachable!()
            };
            *operator = replacement;
            return true;
        }
        _ => {}
    }

    match &mut expression.kind {
        ExpressionKind::Literal(_) | ExpressionKind::Name(_) | ExpressionKind::Symbol(_) => false,
        ExpressionKind::List(values)
        | ExpressionKind::Set(values)
        | ExpressionKind::Tuple(values) => {
            values.iter_mut().any(|value| apply_expression(value, edit))
        }
        ExpressionKind::Map(entries) => entries.iter_mut().any(|entry| {
            apply_expression(&mut entry.key, edit) || apply_expression(&mut entry.value, edit)
        }),
        ExpressionKind::ListComprehension { value, clauses }
        | ExpressionKind::SetComprehension { value, clauses } => {
            apply_expression(value, edit) || apply_clauses(clauses, edit)
        }
        ExpressionKind::MapComprehension {
            key,
            value,
            clauses,
        } => {
            apply_expression(key, edit)
                || apply_expression(value, edit)
                || apply_clauses(clauses, edit)
        }
        ExpressionKind::Mock { cases, fallback } => {
            cases.iter_mut().any(|case| {
                apply_expression(&mut case.call, edit) || apply_expression(&mut case.result, edit)
            }) || apply_expression(fallback, edit)
        }
        ExpressionKind::Lambda { body, .. }
        | ExpressionKind::Async {
            expression: body, ..
        }
        | ExpressionKind::Await { expression: body }
        | ExpressionKind::Throw { error: body }
        | ExpressionKind::Unary { operand: body, .. } => apply_expression(body, edit),
        ExpressionKind::Member { object, .. } => apply_expression(object, edit),
        ExpressionKind::Index { object, index } => {
            apply_expression(object, edit) || apply_expression(index, edit)
        }
        ExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            apply_expression(object, edit)
                || [start, end, step].into_iter().any(|value| {
                    value
                        .as_mut()
                        .is_some_and(|value| apply_expression(value, edit))
                })
        }
        ExpressionKind::TypeApplication { callee, .. } => apply_expression(callee, edit),
        ExpressionKind::Call { callee, arguments } => {
            apply_expression(callee, edit)
                || arguments.iter_mut().any(|argument| {
                    apply_expression(&mut argument.value, edit)
                        || argument
                            .expected_error
                            .as_mut()
                            .is_some_and(|error| apply_expression(error, edit))
                })
        }
        ExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            apply_expression(value, edit)
                || apply_expression(condition, edit)
                || apply_expression(fallback, edit)
        }
        ExpressionKind::Fallback { value, fallback } => {
            apply_expression(value, edit) || apply_expression(fallback, edit)
        }
        ExpressionKind::Binary { left, right, .. } => {
            apply_expression(left, edit) || apply_expression(right, edit)
        }
    }
}

fn apply_clauses(clauses: &mut [ComprehensionClause], edit: MutationEdit) -> bool {
    clauses.iter_mut().any(|clause| {
        apply_expression(&mut clause.iterable, edit)
            || clause
                .condition
                .as_mut()
                .is_some_and(|condition| apply_expression(condition, edit))
    })
}
