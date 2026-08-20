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
    pub bindings: Vec<Binding>,
    pub traits: Vec<TraitDeclaration>,
    pub imports: Vec<ImportDeclaration>,
}
