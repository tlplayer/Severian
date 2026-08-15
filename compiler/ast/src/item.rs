use crate::{Block, Expr, Stmt, Type, TypePath};

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(FunctionDecl),
    Class(ClassDecl),
    Trait(TraitDecl),
    Enum(EnumDecl),
    Import(ImportDecl),
    Statement(Stmt),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Function(node) => node.span,
            Item::Class(node) => node.span,
            Item::Trait(node) => node.span,
            Item::Enum(node) => node.span,
            Item::Import(node) => node.span,
            Item::Statement(node) => node.span(),
        }
    }
}

//
// ===== Source locations =====
//

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn empty(at: usize) -> Self {
        Self { start: at, end: at }
    }

    pub const fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident {
    pub span: Span,
    pub name: String,
}

//
// ===== Declarations =====
//

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub span: Span,
    pub kind: ImportKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportKind {
    Local {
        path: String,
        alias: Option<Ident>,
    },
    Module {
        path: Vec<Ident>,
        alias: Option<Ident>,
    },
    From {
        module: Vec<Ident>,
        names: Vec<ImportName>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportName {
    pub span: Span,
    pub name: Ident,
    pub alias: Option<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub span: Span,
    pub native_symbol: Option<String>,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub generic_params: Vec<GenericParameter>,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub contract: Option<FunctionContract>,
    pub body: Block,
    pub tests: Vec<TestBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParameter {
    pub span: Span,
    pub name: Ident,
    /// Capability constraints are intersected. For example, `T: Numeric +
    /// Float` accepts only floating-point tensor elements.
    pub constraints: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionContract {
    pub span: Span,
    pub clauses: Vec<ContractClause>,
    pub capabilities: Vec<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContractClause {
    pub span: Span,
    pub deferred: bool,
    pub condition: Expr,
    pub failure: Option<ContractFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContractFailure {
    pub span: Span,
    pub message: String,
    pub location: bool,
    pub vars: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decorator {
    pub span: Span,
    pub name: TypePath,
    pub symbols: Vec<DecoratorSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoratorSymbol {
    pub span: Span,
    pub spelling: String,
    /// An optional value for a named semantic policy, as in
    /// `@tensor(backend = auto)`. Positional selectors leave this unset.
    pub value: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub span: Span,
    pub name: Ident,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub span: Span,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub generic_params: Vec<GenericParameter>,
    pub traits: Vec<Type>,
    pub fields: Vec<Field>,
    pub constructors: Vec<ConstructorDecl>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub span: Span,
    pub name: Ident,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
    /// Invariants checked after a complete object value has been assembled.
    ///
    /// A field invariant may refer to any field in the enclosing class.  It is
    /// therefore deliberately not treated as a setter-local predicate: builders,
    /// structural conversions, and transactional updates all validate the same
    /// final object state.
    pub constraints: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitDecl {
    pub span: Span,
    pub name: Ident,
    pub generic_params: Vec<GenericParameter>,
    /// Semantic decorators declared by this trait. Unlike function
    /// decorators, these define a semantic namespace rather than activating
    /// it for executable code.
    pub decorators: Vec<Decorator>,
    /// Traits named directly in this contract. Their requirements are part of
    /// this trait transitively; Severian deliberately has no separate
    /// `extends` or `inherits` syntax.
    pub composed_traits: Vec<Type>,
    /// Compile-time metadata every concrete implementation contributes to the
    /// trait's closed implementation registry. A default turns a requirement
    /// into an inherited contribution that implementations may override.
    pub properties: Vec<TraitProperty>,
    pub methods: Vec<TraitMethod>,
    pub operators: Vec<TraitOperator>,
    /// Compiler-owned behavior executed on entry to and exit from a semantic
    /// scope. A provider must declare one `with` and one `without` body.
    pub scoped_behaviors: Vec<TraitScopedBehavior>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitProperty {
    pub span: Span,
    pub name: Ident,
    pub ty: Type,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitScopedBehaviorPhase {
    With,
    Without,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitScopedBehavior {
    pub span: Span,
    pub phase: TraitScopedBehaviorPhase,
    pub params: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub span: Span,
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub span: Span,
    pub name: Ident,
    pub fields: Vec<Parameter>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstructorDecl {
    pub span: Span,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub params: Vec<Parameter>,
    pub contract: Option<FunctionContract>,
    pub body: Block,
    pub tests: Vec<TestBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestBlock {
    pub span: Span,
    pub modes: Vec<TestMode>,
    pub name: Option<Ident>,
    pub return_type: Option<Type>,
    pub contract: Option<FunctionContract>,
    pub body: Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    Property,
    Bench,
    Chaos,
    Integration,
    Profile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub span: Span,
    pub name: Ident,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitOperator {
    pub span: Span,
    pub symbol: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
}

//
// ===== Statements =====
//
