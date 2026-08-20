#![forbid(unsafe_code)]

mod expression;
mod statement;
mod types;

pub use expression::{Expression, ExpressionKind};
pub use statement::Binding;
pub use types::TypeAnnotation;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub bindings: Vec<Binding>,
}
