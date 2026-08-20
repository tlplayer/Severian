use severian_diagnostics::Diagnostic;
use severian_hir::{BindingId, Expression, ExpressionKind, Module, Program};
use std::collections::BTreeSet;

pub fn validate(program: &Program) -> Result<(), Diagnostic> {
    for module in &program.modules {
        validate_module(module)?;
    }
    Ok(())
}

fn validate_module(module: &Module) -> Result<(), Diagnostic> {
    let mut declared = BTreeSet::new();
    for binding in &module.bindings {
        validate_expression(&binding.value, &declared)?;
        declared.insert(binding.id);
    }
    Ok(())
}

fn validate_expression(
    expression: &Expression,
    declared: &BTreeSet<BindingId>,
) -> Result<(), Diagnostic> {
    match &expression.kind {
        ExpressionKind::Literal(_) => Ok(()),
        ExpressionKind::Binding(id) if declared.contains(id) => Ok(()),
        ExpressionKind::Binding(_) => Err(Diagnostic::new(
            "E000301",
            "value used before it is available",
            Some(expression.span),
        )),
        ExpressionKind::Unary { operand, .. } => validate_expression(operand, declared),
        ExpressionKind::Binary { left, right, .. } => {
            validate_expression(left, declared)?;
            validate_expression(right, declared)
        }
    }
}
