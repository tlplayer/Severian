use crate::Expression;
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub value: Expression,
    pub span: Span,
}
