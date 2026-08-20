#![forbid(unsafe_code)]

#[path = "analyzer/expression/mod.rs"]
mod expression;

use severian_ast::Module as AstModule;
use severian_diagnostics::Diagnostic;
use severian_hir::{Binding, HirId, Module, Program, TypeKind, TypeTable};
use std::collections::BTreeMap;

pub fn analyze(ast: &AstModule) -> Result<Program, Diagnostic> {
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
        if let Some(annotation) = &ast_binding.annotation {
            let primitive_name = primitive.path.rsplit('.').next().unwrap_or(primitive.path);
            if annotation.name != primitive_name {
                return Err(Diagnostic::new(
                    "E000204",
                    format!("unknown type `{}`", annotation.name),
                    Some(annotation.span),
                ));
            }
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
    Ok(Program {
        modules: vec![Module { bindings }],
        types,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_ast::{Binding as AstBinding, Expression as AstExpression, ExpressionKind};
    use severian_source::{SourceId, Span};
    #[test]
    fn annotation_literals_and_addition_share_type_identity() {
        let span = Span::new(SourceId(0), 0, 1);
        let ast = AstModule {
            bindings: vec![
                AstBinding {
                    name: "b".into(),
                    annotation: Some(severian_ast::TypeAnnotation {
                        name: "int".into(),
                        span,
                    }),
                    value: AstExpression {
                        kind: ExpressionKind::Integer(2),
                        span,
                    },
                    span,
                },
                AstBinding {
                    name: "a".into(),
                    annotation: None,
                    value: AstExpression {
                        kind: ExpressionKind::Add {
                            left: Box::new(AstExpression {
                                kind: ExpressionKind::Integer(1),
                                span,
                            }),
                            right: Box::new(AstExpression {
                                kind: ExpressionKind::Name("b".into()),
                                span,
                            }),
                        },
                        span,
                    },
                    span,
                },
            ],
        };
        let program = analyze(&ast).unwrap();
        let bindings = &program.modules[0].bindings;
        let integer = bindings[0].value.type_id;
        assert_eq!(bindings[1].value.type_id, integer);
        let severian_hir::ExpressionKind::Add { left, right } = &bindings[1].value.kind else {
            panic!("second binding must remain an addition");
        };
        assert_eq!(left.type_id, integer);
        assert_eq!(right.type_id, integer);
        assert_eq!(
            program.types.get(integer),
            Some(&TypeKind::Primitive(primitive_id()))
        );
    }

    fn primitive_id() -> severian_interface::PrimitiveId {
        severian_primitives::default_integer().unwrap().id
    }
}
