use crate::{Expression, TypeAnnotation};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub annotation: Option<TypeAnnotation>,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Binding(Binding),
    Expression(Expression),
}
