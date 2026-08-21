use crate::{BindingId, Expression, TypeId};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub id: BindingId,
    pub type_id: TypeId,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Binding(BindingId),
    Expression(Expression),
    Return(Option<Expression>),
    Assert {
        condition: Expression,
        message: Option<Expression>,
        span: Span,
        condition_span: Span,
    },
    If {
        condition: Expression,
        then_block: Block,
        else_block: Block,
    },
    Match {
        subject: Expression,
        arms: Vec<MatchArm>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub binding: Option<BindingId>,
    pub type_id: Option<TypeId>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub statements: Vec<Statement>,
}
