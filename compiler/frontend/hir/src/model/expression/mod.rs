use crate::{HirId, TypeId};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub id: HirId,
    pub type_id: TypeId,
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Integer(i64),
    Binding(HirId),
    Add {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}
