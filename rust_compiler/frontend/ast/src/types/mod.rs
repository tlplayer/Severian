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

impl Decorator {
    /// Compiler policies select an IR/backend lowering route. They are not
    /// foreign-language attributes and must never enter ABI resolution.
    pub fn is_compile_policy(&self) -> bool {
        matches!(
            self.name.as_str(),
            "compile" | "mlir" | "stablehlo" | "xla" | "triton"
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionParameter {
    pub name: String,
    pub annotation: TypeAnnotation,
    pub variadic: bool,
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
    pub parameters: Vec<String>,
    pub cases: Vec<Vec<crate::Expression>>,
    pub modes: Vec<String>,
    pub contracts: Vec<FunctionContract>,
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
    VariadicPack {
        parameter: String,
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
    /// A compile-time dimension value used as a tensor generic argument.
    /// This is deliberately not a named type: dimensions never enter the
    /// ordinary type interner as synthetic `TypeId`s.
    DimensionConstant(u64),
    /// A dimension whose value is supplied by the launch/storage descriptor,
    /// but whose axis identity is already known at compile time.
    DimensionRuntime(u32),
    /// A variadic shape argument such as `*Shape` in `Tensor[T, *Shape]`.
    /// The referenced generic parameter is kinded as `Shape` during semantic
    /// analysis and expands to zero or more `DimExpr` values.
    ShapeSpread(String),
    Function {
        parameters: Vec<TypeAnnotation>,
        result: Box<TypeAnnotation>,
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
            TypeAnnotationKind::DimensionConstant(_)
            | TypeAnnotationKind::DimensionRuntime(_)
            | TypeAnnotationKind::ShapeSpread(_)
            | TypeAnnotationKind::Function { .. }
            | TypeAnnotationKind::Union(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorSyntax {
    Index,
    If,
    Else,
    Pipe,
    BitwiseAnd,
    BitwiseXor,
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
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub parameters: Vec<OperatorParameter>,
    pub result: TypeAnnotation,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorImplementation {
    pub decorators: Vec<Decorator>,
    pub operator: OperatorSyntax,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub parameters: Vec<OperatorParameter>,
    pub contracts: Vec<FunctionContract>,
    pub result: TypeAnnotation,
    pub body: Vec<crate::Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub decorators: Vec<Decorator>,
    /// Standalone decorators in the trait body expose its semantic members.
    pub namespaces: Vec<Decorator>,
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
    /// A trailing `:` after the implemented traits declares that this source
    /// class completes a compiler-owned primitive rather than introducing a
    /// second nominal type.
    pub primitive: bool,
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
    pub traits: Vec<TypeAnnotation>,
    pub fields: Vec<PropertyDeclaration>,
    pub constructors: Vec<FunctionDeclaration>,
    pub methods: Vec<FunctionDeclaration>,
    pub operators: Vec<OperatorImplementation>,
    pub tests: Vec<crate::TestDeclaration>,
    pub span: Span,
}

/// Behavior added to an existing type without changing that type's identity.
/// Extensions may contain executable members only; semantic analysis rejects
/// every member that is already defined directly by the target type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionDeclaration {
    pub decorators: Vec<Decorator>,
    pub target: TypeAnnotation,
    pub methods: Vec<FunctionDeclaration>,
    pub operators: Vec<OperatorImplementation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<PropertyDeclaration>,
    pub accepted_values: Vec<crate::Literal>,
    pub transitions: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDeclaration {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}
