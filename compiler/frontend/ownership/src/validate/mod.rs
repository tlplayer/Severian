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
        Statement::Binding(id) => {
            let binding = bindings
                .get(id)
                .expect("HIR statement references a binding");
            validate_expression(&binding.value, declared)?;
            declared.insert(*id);
            Ok(())
        }
        Statement::Expression(expression) => validate_expression(expression, declared),
    }
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
        ExpressionKind::Call { arguments, .. } => {
            for argument in arguments {
                validate_expression(argument, declared)?;
            }
            Ok(())
        }
        ExpressionKind::Unary { operand, .. } => validate_expression(operand, declared),
        ExpressionKind::Binary { left, right, .. } => {
            validate_expression(left, declared)?;
            validate_expression(right, declared)
        }
    }
}
