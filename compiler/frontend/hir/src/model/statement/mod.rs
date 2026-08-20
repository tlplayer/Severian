use crate::{Expression, HirId};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub id: HirId,
    pub value: Expression,
    pub span: Span,
}
