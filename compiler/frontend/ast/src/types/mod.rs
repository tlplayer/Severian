use crate::Expression;
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
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub parameters: Vec<FunctionParameter>,
    pub result: TypeAnnotation,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub definition: Option<TypeAnnotation>,
    pub span: Span,
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
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyDeclaration {
    pub name: String,
    pub annotation: TypeAnnotation,
    pub default: Option<Expression>,
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
    pub operator: OperatorSyntax,
    pub parameters: Vec<OperatorParameter>,
    pub result: TypeAnnotation,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub type_parameters: Vec<String>,
    pub bases: Vec<TypeAnnotation>,
    pub properties: Vec<PropertyDeclaration>,
    pub operators: Vec<OperatorDeclaration>,
    pub span: Span,
}
