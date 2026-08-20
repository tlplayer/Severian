#![forbid(unsafe_code)]

#[path = "model/expression/mod.rs"]
mod expression;
#[path = "model/statement/mod.rs"]
mod statement;

pub use expression::{Expression, ExpressionKind};
pub use severian_universal::TypeId;
pub use statement::Binding;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Program {
    pub modules: Vec<Module>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub bindings: Vec<Binding>,
}
