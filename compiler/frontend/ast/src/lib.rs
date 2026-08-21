#![forbid(unsafe_code)]

mod expression;
mod statement;
mod types;

pub use expression::{BinaryOperator, Expression, ExpressionKind, Literal, UnaryOperator};
pub use statement::{Binding, Statement};
pub use types::{
    Decorator, DecoratorArgument, DecoratorValue, FunctionDeclaration, FunctionParameter,
    ImportDeclaration, ImportSubject, OperatorDeclaration, OperatorParameter, OperatorSyntax,
    PropertyDeclaration, TraitDeclaration, TypeAnnotation, TypeAnnotationKind, TypeDeclaration,
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
    Expression(Expression),
    Function(FunctionDeclaration),
    Type(TypeDeclaration),
}
