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
pub struct MatchCase {
    pub binding: Option<String>,
    pub annotation: Option<TypeAnnotation>,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Binding(Binding),
    Expression(Expression),
    Return {
        value: Option<Expression>,
        span: Span,
    },
    Assert {
        condition: Expression,
        message: Option<Expression>,
        span: Span,
    },
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Vec<Statement>,
        span: Span,
    },
    Match {
        subject: Expression,
        cases: Vec<MatchCase>,
        span: Span,
    },
}
