use severian_diagnostics::Diagnostic;
use severian_hir::{BindingId, Expression, ExpressionKind, Module, Program, Statement};
use std::collections::{BTreeMap, BTreeSet};

pub fn validate(program: &Program) -> Result<(), Diagnostic> {
    for module in &program.modules {
        validate_module(module)?;
    }
    Ok(())
}

fn validate_module(module: &Module) -> Result<(), Diagnostic> {
    let mut declared = BTreeSet::new();
    let bindings = module
        .bindings
        .iter()
        .map(|binding| (binding.id, binding))
        .collect::<BTreeMap<_, _>>();
    if module.initializer.statements.is_empty()
        && module
            .functions
            .iter()
            .all(|function| function.body.is_none())
    {
        for binding in &module.bindings {
            validate_expression(&binding.value, &declared)?;
            declared.insert(binding.id);
        }
        return Ok(());
    }
    for statement in &module.initializer.statements {
        validate_statement(statement, &bindings, &mut declared)?;
    }
    let globals = declared.clone();
    for function in &module.functions {
        let Some(body) = &function.body else {
            continue;
        };
        declared.clone_from(&globals);
        declared.extend(
            function
                .parameters
                .iter()
                .map(|parameter| parameter.binding),
        );
        for statement in &body.statements {
            validate_statement(statement, &bindings, &mut declared)?;
        }
    }
    Ok(())
}

fn validate_statement(
    statement: &Statement,
    bindings: &BTreeMap<BindingId, &severian_hir::Binding>,
    declared: &mut BTreeSet<BindingId>,
) -> Result<(), Diagnostic> {
    match statement {
        Statement::Sequence(block) => {
            for statement in &block.statements {
                validate_statement(statement, bindings, declared)?;
            }
            Ok(())
        }
        Statement::FieldUpdate { binding, value, .. }
        | Statement::FieldSet { binding, value, .. } => {
            if !declared.contains(binding) {
                return Err(Diagnostic::new(
                    "E000301",
                    "class value updated before it is available",
                    Some(value.span),
                ));
            }
            validate_expression(value, declared)
        }
        Statement::Binding(id) => {
            let binding = bindings
                .get(id)
                .expect("HIR statement references a binding");
            validate_expression(&binding.value, declared)?;
            declared.insert(*id);
            Ok(())
        }
        Statement::Expression(expression) => validate_expression(expression, declared),
        Statement::Return(value) => value
            .as_ref()
            .map_or(Ok(()), |value| validate_expression(value, declared)),
        Statement::Assert {
            condition, message, ..
        } => {
            validate_expression(condition, declared)?;
            if let Some(message) = message {
                validate_expression(message, declared)?;
            }
            Ok(())
        }
        Statement::ExpectThrow { body, .. } => {
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            Ok(())
        }
        Statement::Try {
            body,
            catch_binding,
            catch_body,
            ..
        } => {
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            let mut catch_declared = declared.clone();
            catch_declared.insert(*catch_binding);
            for statement in &catch_body.statements {
                validate_statement(statement, bindings, &mut catch_declared)?;
            }
            Ok(())
        }
        Statement::If {
            condition,
            then_block,
            else_block,
        } => {
            validate_expression(condition, declared)?;
            let mut then_declared = declared.clone();
            for statement in &then_block.statements {
                validate_statement(statement, bindings, &mut then_declared)?;
            }
            let mut else_declared = declared.clone();
            for statement in &else_block.statements {
                validate_statement(statement, bindings, &mut else_declared)?;
            }
            Ok(())
        }
        Statement::While {
            condition, body, ..
        } => {
            validate_expression(condition, declared)?;
            let mut body_declared = declared.clone();
            for statement in &body.statements {
                validate_statement(statement, bindings, &mut body_declared)?;
            }
            Ok(())
        }
        Statement::Break { .. } | Statement::Continue { .. } => Ok(()),
        Statement::Match { subject, arms } => {
            validate_expression(subject, declared)?;
            for arm in arms {
                let mut arm_declared = declared.clone();
                if let Some(binding) = arm.binding {
                    arm_declared.insert(binding);
                }
                for statement in &arm.body.statements {
                    validate_statement(statement, bindings, &mut arm_declared)?;
                }
            }
            Ok(())
        }
    }
}

fn validate_expression(
    expression: &Expression,
    declared: &BTreeSet<BindingId>,
) -> Result<(), Diagnostic> {
    match &expression.kind {
        ExpressionKind::Aggregate { fields, .. } => {
            for field in fields {
                validate_expression(field, declared)?;
            }
            Ok(())
        }
        ExpressionKind::Field { object, .. } => validate_expression(object, declared),
        ExpressionKind::Literal(_) | ExpressionKind::Function(_) => Ok(()),
        ExpressionKind::Binding(id) if declared.contains(id) => Ok(()),
        ExpressionKind::Binding(_) => Err(Diagnostic::new(
            "E000301",
            "value used before it is available",
            Some(expression.span),
        )),
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                validate_expression(argument, declared)?;
            }
            Ok(())
        }
        ExpressionKind::Async { expression, .. } | ExpressionKind::Await(expression) => {
            validate_expression(expression, declared)
        }
        ExpressionKind::AsyncFieldUpdate { binding, value, .. } => {
            if !declared.contains(binding) {
                return Err(Diagnostic::new(
                    "E000301",
                    "class value updated before it is available",
                    Some(expression.span),
                ));
            }
            validate_expression(value, declared)
        }
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            validate_expression(condition, declared)?;
            validate_expression(value, declared)?;
            validate_expression(fallback, declared)
        }
        ExpressionKind::Throw(error) => validate_expression(error, declared),
        ExpressionKind::Unary { operand, .. }
        | ExpressionKind::Borrow { operand, .. }
        | ExpressionKind::Move(operand)
        | ExpressionKind::Convert { operand, .. } => validate_expression(operand, declared),
        ExpressionKind::Binary { left, right, .. } => {
            validate_expression(left, declared)?;
            validate_expression(right, declared)
        }
    }
}
