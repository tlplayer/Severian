#![forbid(unsafe_code)]

mod build;
#[path = "model/operation/mod.rs"]
mod operation;
#[path = "model/value/mod.rs"]
mod value;

pub use build::build;
pub use operation::Operation;
use severian_hir::HirId;
pub use value::{Value, ValueId};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub operations: Vec<Operation>,
    pub bindings: Vec<(HirId, ValueId)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{Binding, Expression, ExpressionKind, HirId, TypeId, TypeTable};
    use severian_source::{SourceId, Span};

    #[test]
    fn builds_constant_then_add() {
        let span = Span::new(SourceId(0), 0, 1);
        let integer = TypeId(0);
        let first = HirId(1);
        let hir = severian_hir::Module {
            types: TypeTable::default(),
            bindings: vec![
                Binding {
                    id: first,
                    span,
                    value: Expression {
                        id: HirId(0),
                        type_id: integer,
                        kind: ExpressionKind::Integer(2),
                        span,
                    },
                },
                Binding {
                    id: HirId(4),
                    span,
                    value: Expression {
                        id: HirId(3),
                        type_id: integer,
                        span,
                        kind: ExpressionKind::Add {
                            left: Box::new(Expression {
                                id: HirId(2),
                                type_id: integer,
                                kind: ExpressionKind::Integer(1),
                                span,
                            }),
                            right: Box::new(Expression {
                                id: HirId(3),
                                type_id: integer,
                                kind: ExpressionKind::Binding(first),
                                span,
                            }),
                        },
                    },
                },
            ],
        };
        assert!(matches!(
            build(&hir).operations.last(),
            Some(Operation::AddInt { .. })
        ));
    }
}
