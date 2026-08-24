use crate::{Expression, Statement};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportDeclaration {
    pub subject: ImportSubject,
    pub source: Option<String>,
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportSubject {
    Name(String),
    Locator(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoratorValue {
    Name(String),
    String(String),
    Integer(String),
    Boolean(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoratorArgument {
    pub name: Option<String>,
    pub value: DecoratorValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decorator {
    pub name: String,
    pub arguments: Vec<DecoratorArgument>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionParameter {
    pub name: String,
    pub annotation: TypeAnnotation,
    pub default: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    /// Executable entry and exit conditions, kept separate from generic
    /// constraints because they survive into runtime IR.
    pub contracts: Vec<FunctionContract>,
    /// Structured interception implemented by this function. This is distinct
    /// from contracts: `with context` names the hook context, while
    /// `with { ... }` declares callable predicates.
    pub hook: Option<HookSpecification>,
    pub parameters: Vec<FunctionParameter>,
    pub result: TypeAnnotation,
    /// `None` denotes an interface declaration. Source functions have an
    /// ordered body, including an explicitly empty body.
    pub body: Option<Vec<Statement>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookSpecification {
    pub context: String,
    pub with_phase: Vec<Statement>,
    pub without_phase: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionContract {
    pub condition: Expression,
    pub deferred: bool,
    pub failure: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub definition: Option<TypeAnnotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestDeclaration {
    pub name: Option<String>,
    pub modes: Vec<String>,
    pub body: Vec<crate::Statement>,
    pub compiler_cases: Vec<CompilerTestCase>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerTestCase {
    pub expectation: CompilerExpectation,
    pub diagnostic_name: Option<String>,
    pub items: Vec<crate::Item>,
    pub body: Vec<crate::Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerExpectation {
    Accept,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenericConstraint {
    Parameter {
        parameter: String,
        bound: TypeAnnotation,
        span: Span,
    },
    Predicate(Expression),
}

/// The one source-level type node used in every annotation position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAnnotation {
    pub kind: TypeAnnotationKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeAnnotationKind {
    Named {
        name: String,
        arguments: Vec<TypeAnnotation>,
    },
    Union(Vec<TypeAnnotation>),
}

impl TypeAnnotation {
    pub fn named(name: impl Into<String>, arguments: Vec<Self>, span: Span) -> Self {
        Self {
            kind: TypeAnnotationKind::Named {
                name: name.into(),
                arguments,
            },
            span,
        }
    }

    pub fn simple_name(&self) -> Option<&str> {
        match &self.kind {
            TypeAnnotationKind::Named { name, arguments } if arguments.is_empty() => Some(name),
            _ => None,
        }
    }

    pub fn named_parts(&self) -> Option<(&str, &[Self])> {
        match &self.kind {
            TypeAnnotationKind::Named { name, arguments } => Some((name, arguments)),
            TypeAnnotationKind::Union(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSyntax {
    Pipe,
    Plus,
    Minus,
    Multiply,
    Divide,
    Remainder,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Contains,
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyConstraint {
    pub condition: Expression,
    /// Present for rejection rules (`invalid -> error`). Absent predicates
    /// are invariants that must evaluate to true.
    pub failure: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyDeclaration {
    pub name: String,
    pub annotation: TypeAnnotation,
    pub default: Option<Expression>,
    /// Ordered validation rules applied whenever a value is stored.
    pub constraints: Vec<PropertyConstraint>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorParameter {
    pub name: String,
    pub annotation: TypeAnnotation,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorDeclaration {
    pub decorators: Vec<Decorator>,
    pub operator: OperatorSyntax,
    pub parameters: Vec<OperatorParameter>,
    pub result: TypeAnnotation,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorImplementation {
    pub decorators: Vec<Decorator>,
    pub operator: OperatorSyntax,
    pub parameters: Vec<OperatorParameter>,
    pub contracts: Vec<FunctionContract>,
    pub result: TypeAnnotation,
    pub body: Vec<crate::Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub decorators: Vec<Decorator>,
    /// Standalone decorators in the trait body define composed hook namespaces.
    pub hook_namespaces: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub bases: Vec<TypeAnnotation>,
    pub properties: Vec<PropertyDeclaration>,
    pub methods: Vec<FunctionDeclaration>,
    pub operators: Vec<OperatorDeclaration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub traits: Vec<TypeAnnotation>,
    pub fields: Vec<PropertyDeclaration>,
    pub constructors: Vec<FunctionDeclaration>,
    pub methods: Vec<FunctionDeclaration>,
    pub operators: Vec<OperatorImplementation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<PropertyDeclaration>,
    pub transitions: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDeclaration {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}
