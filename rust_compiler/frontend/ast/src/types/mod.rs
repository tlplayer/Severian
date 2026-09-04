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
            "compile" | "mlir" | "stablehlo" | "xla" | "triton" | "unsafe"
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
    /// `true` for a leading-`->` compile-time specialization declaration.
    /// Its generic body is checked after the call-site types are substituted.
    pub compile_time: bool,
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
    pub matrix: bool,
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

const fn operator_hash(value: &str) -> u128 {
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let bytes = value.as_bytes();
    let mut index = 0;
    let mut result = OFFSET;
    while index < bytes.len() {
        result ^= bytes[index] as u128;
        result = result.wrapping_mul(PRIME);
        index += 1;
    }
    result
}

/// Stable, open identity of operator syntax.
///
/// Individual operators deliberately are associated constants rather than enum
/// variants. Parsers may construct an identity for any source spelling with
/// `from_spelling`; known constants only document the standard prelude.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperatorSyntax(pub u128);

#[allow(non_upper_case_globals)]
impl OperatorSyntax {
    pub const Index: Self = Self(operator_hash("[]"));
    pub const If: Self = Self(operator_hash("if"));
    pub const Else: Self = Self(operator_hash("else"));
    pub const Pipe: Self = Self(operator_hash("|"));
    pub const BitwiseAnd: Self = Self(operator_hash("&"));
    pub const BitwiseXor: Self = Self(operator_hash("^"));
    pub const Plus: Self = Self(operator_hash("+"));
    pub const Add: Self = Self::Plus;
    pub const Minus: Self = Self(operator_hash("-"));
    pub const Subtract: Self = Self::Minus;
    pub const Multiply: Self = Self(operator_hash("*"));
    pub const Divide: Self = Self(operator_hash("/"));
    pub const FloorDivide: Self = Self(operator_hash("//"));
    pub const Remainder: Self = Self(operator_hash("%"));
    pub const Power: Self = Self(operator_hash("**"));
    pub const ShiftLeft: Self = Self(operator_hash("<<"));
    pub const ShiftRight: Self = Self(operator_hash(">>"));
    pub const Conversion: Self = Self(operator_hash("<=>"));
    pub const Equal: Self = Self(operator_hash("=="));
    // Identity intentionally has a semantic identity distinct from equality,
    // even though its legacy surface spelling is also `==`.
    pub const Identity: Self = Self(operator_hash("compiler.identity"));
    pub const NotEqual: Self = Self(operator_hash("!="));
    pub const Less: Self = Self(operator_hash("<"));
    pub const LessEqual: Self = Self(operator_hash("<="));
    pub const Greater: Self = Self(operator_hash(">"));
    pub const GreaterEqual: Self = Self(operator_hash(">="));
    pub const Contains: Self = Self(operator_hash("in"));
    pub const And: Self = Self(operator_hash("and"));
    pub const Or: Self = Self(operator_hash("or"));
    pub const Not: Self = Self(operator_hash("not"));

    pub const fn from_spelling(spelling: &str) -> Self {
        Self(operator_hash(spelling))
    }

    pub const fn stable_id(self) -> u128 {
        self.0
    }

    pub fn standard_spelling(self) -> Option<&'static str> {
        if self == Self::Index {
            Some("[]")
        } else if self == Self::If {
            Some("if")
        } else if self == Self::Else {
            Some("else")
        } else if self == Self::Pipe {
            Some("|")
        } else if self == Self::BitwiseAnd {
            Some("&")
        } else if self == Self::BitwiseXor {
            Some("^")
        } else if self == Self::Plus {
            Some("+")
        } else if self == Self::Minus {
            Some("-")
        } else if self == Self::Multiply {
            Some("*")
        } else if self == Self::Divide {
            Some("/")
        } else if self == Self::FloorDivide {
            Some("//")
        } else if self == Self::Remainder {
            Some("%")
        } else if self == Self::Power {
            Some("**")
        } else if self == Self::ShiftLeft {
            Some("<<")
        } else if self == Self::ShiftRight {
            Some(">>")
        } else if self == Self::Conversion {
            Some("<=>")
        } else if self == Self::Equal || self == Self::Identity {
            Some("==")
        } else if self == Self::NotEqual {
            Some("!=")
        } else if self == Self::Less {
            Some("<")
        } else if self == Self::LessEqual {
            Some("<=")
        } else if self == Self::Greater {
            Some(">")
        } else if self == Self::GreaterEqual {
            Some(">=")
        } else if self == Self::Contains {
            Some("in")
        } else if self == Self::And {
            Some("and")
        } else if self == Self::Or {
            Some("or")
        } else if self == Self::Not {
            Some("not")
        } else {
            None
        }
    }
}

impl std::fmt::Debug for OperatorSyntax {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.standard_spelling() {
            Some(spelling) => formatter.debug_tuple("Operator").field(&spelling).finish(),
            None => formatter
                .debug_tuple("Operator")
                .field(&format_args!("{:032x}", self.0))
                .finish(),
        }
    }
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
    /// Optional compiler-symbol tag supplied as the first `[Tag: Y]`
    /// parameter. It names the semantic operation independently of spelling.
    pub tag: Option<String>,
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
    /// Optional compiler-symbol tag supplied as the first `[Tag: Y]`
    /// parameter. It names the semantic operation independently of spelling.
    pub tag: Option<String>,
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
    pub type_parameter_defaults: Vec<Option<TypeAnnotation>>,
    pub constraints: Vec<GenericConstraint>,
    pub traits: Vec<TypeAnnotation>,
    /// Alternate source spellings that resolve to this class identity. The
    /// primitive pointer declaration uses this to connect `pointer[T]` with
    /// the structural `*[T]` spelling without introducing another type.
    pub aliases: Vec<TypeAnnotation>,
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
    pub type_parameters: Vec<String>,
    pub constraints: Vec<GenericConstraint>,
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
