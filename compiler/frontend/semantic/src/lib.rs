#![forbid(unsafe_code)]

#[path = "analyzer/expression/mod.rs"]
mod expression;

use severian_ast::Module as AstModule;
use severian_diagnostics::Diagnostic;
use severian_hir::{Binding, HirId, Module, TypeKind, TypeTable};
use std::collections::BTreeMap;

pub fn analyze(ast: &AstModule) -> Result<Module, Diagnostic> {
    let primitive = severian_primitives::default_integer().map_err(|error| {
        Diagnostic::new(
            "E000250",
            format!("primitive bootstrap failed: {error:?}"),
            None,
        )
    })?;
    let mut types = TypeTable::default();
    let integer = types.intern(TypeKind::Primitive(primitive.id));
    let mut names = BTreeMap::new();
    let mut bindings = Vec::new();
    let mut next_id = 0;
    for ast_binding in &ast.bindings {
        if names.contains_key(&ast_binding.name) {
            return Err(Diagnostic::new(
                "E000203",
                format!("binding `{}` is already defined", ast_binding.name),
                Some(ast_binding.span),
            ));
        }
        let value = expression::analyze(&ast_binding.value, integer, &names, &mut next_id)?;
        let id = HirId(next_id);
        next_id += 1;
        names.insert(ast_binding.name.clone(), (id, integer));
        bindings.push(Binding {
            id,
            value,
            span: ast_binding.span,
        });
    }
    Ok(Module { bindings, types })
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_ast::{Binding as AstBinding, Expression as AstExpression, ExpressionKind};
    use severian_source::{SourceId, Span};
    #[test]
    fn resolves_prior_integer_binding() {
        let span = Span::new(SourceId(0), 0, 1);
        let ast = AstModule {
            bindings: vec![
                AstBinding {
                    name: "b".into(),
                    value: AstExpression {
                        kind: ExpressionKind::Integer(2),
                        span,
                    },
                    span,
                },
                AstBinding {
                    name: "a".into(),
                    value: AstExpression {
                        kind: ExpressionKind::Name("b".into()),
                        span,
                    },
                    span,
                },
            ],
        };
        assert_eq!(analyze(&ast).unwrap().bindings.len(), 2);
    }
}
