#![forbid(unsafe_code)]

mod expression;
mod statement;
mod types;

pub use expression::{
    BinaryOperator, CallArgument, Expression, ExpressionKind, Literal, MapEntry, MockCase,
    TaskOwner, UnaryOperator,
};
pub use statement::{Binding, LoopGuard, LoopGuardAction, MatchCase, SelectCase, Statement};
pub use types::{
    ClassDeclaration, CompilerExpectation, CompilerTestCase, Decorator, DecoratorArgument,
    DecoratorValue, EnumDeclaration, EnumVariant, FunctionContract, FunctionDeclaration,
    FunctionParameter, GenericConstraint, HookSpecification, ImportDeclaration, ImportSubject,
    OperatorDeclaration, OperatorImplementation, OperatorParameter, OperatorSyntax,
    PropertyConstraint, PropertyDeclaration, TestDeclaration, TraitDeclaration, TypeAnnotation,
    TypeAnnotationKind, TypeDeclaration,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Import(ImportDeclaration),
    Trait(TraitDeclaration),
    Class(ClassDeclaration),
    Enum(EnumDeclaration),
    Binding(Binding),
    Expression(Expression),
    Function(FunctionDeclaration),
    Type(TypeDeclaration),
    Test(TestDeclaration),
}
