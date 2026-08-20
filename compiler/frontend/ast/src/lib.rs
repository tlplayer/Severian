#![forbid(unsafe_code)]

mod expression;
mod statement;
mod types;

pub use expression::{BinaryOperator, Expression, ExpressionKind, Literal, UnaryOperator};
pub use statement::Binding;
pub use types::{
    ImportDeclaration, OperatorDeclaration, OperatorParameter, OperatorSyntax, PropertyDeclaration,
    TraitDeclaration, TypeAnnotation, TypeAnnotationKind,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Import(ImportDeclaration),
    Trait(TraitDeclaration),
    Binding(Binding),
}
