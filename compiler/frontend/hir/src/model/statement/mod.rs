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
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub statements: Vec<Statement>,
}
