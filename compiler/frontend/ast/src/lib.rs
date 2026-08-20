#![forbid(unsafe_code)]

mod expression;
mod statement;

pub use expression::{Expression, ExpressionKind};
pub use statement::Binding;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub bindings: Vec<Binding>,
}
