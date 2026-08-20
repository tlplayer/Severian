use severian_ast::{Expression as AstExpression, ExpressionKind as AstExpressionKind};
use severian_diagnostics::Diagnostic;
use severian_hir::{Expression, ExpressionKind, HirId, TypeId};
use std::collections::BTreeMap;

pub fn analyze(
    expression: &AstExpression,
    integer: TypeId,
    bindings: &BTreeMap<String, (HirId, TypeId)>,
    next_id: &mut u32,
) -> Result<Expression, Diagnostic> {
    let id = HirId(*next_id);
    *next_id += 1;
    let kind = match &expression.kind {
        AstExpressionKind::Integer(value) => ExpressionKind::Integer(*value),
        AstExpressionKind::Name(name) => {
            let Some((binding, binding_type)) = bindings.get(name) else {
                return Err(Diagnostic::new(
                    "E000201",
                    format!("unknown binding `{name}`"),
                    Some(expression.span),
                ));
            };
            if *binding_type != integer {
                return Err(Diagnostic::new(
                    "E000202",
                    "addition requires int operands",
                    Some(expression.span),
                ));
            }
            ExpressionKind::Binding(*binding)
        }
        AstExpressionKind::Add { left, right } => {
            let left = analyze(left, integer, bindings, next_id)?;
            let right = analyze(right, integer, bindings, next_id)?;
            if left.type_id != integer || right.type_id != integer {
                return Err(Diagnostic::new(
                    "E000202",
                    "addition requires int operands",
                    Some(expression.span),
                ));
            }
            ExpressionKind::Add {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    };
    Ok(Expression {
        id,
        type_id: integer,
        kind,
        span: expression.span,
    })
}
