use crate::{BindingId, FunctionId, HirId, TypeId};
use severian_source::Span;
use severian_universal::{BinaryOperator, LiteralValue, UnaryOperator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub id: HirId,
    pub type_id: TypeId,
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Literal(LiteralValue),
    Binding(BindingId),
    Call {
        function: FunctionId,
        arguments: Vec<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}
