#![forbid(unsafe_code)]

#[path = "model/expression/mod.rs"]
mod expression;
#[path = "model/statement/mod.rs"]
mod statement;

pub use expression::{Callee, Conversion, ConversionKind};
pub use expression::{Expression, ExpressionKind};
pub use severian_universal::{
    CompileRoute, CompilerId, DefId, GenericParamId, OpId, Substitution, TypeId,
};
pub use statement::{Binding, Block, MatchArm, Statement};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(pub u32);

/// Stable identity of a mutable lexical variable. A binding identifies one
/// source assignment; every assignment to the same variable shares this ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariantId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterfaceId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FfiId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbiId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderId(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallType {
    Severian,
    External(ExternalCall),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalCall {
    pub interface: InterfaceId,
    pub symbol: SymbolId,
    pub provider: Option<ProviderId>,
    pub ffi: FfiId,
    pub abi: AbiId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryModifier {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryType {
    pub ty: TypeId,
    pub modifiers: Vec<BoundaryModifier>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionParameter {
    pub binding: BindingId,
    pub name: String,
    pub contract: BoundaryType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDeclaration {
    pub id: FunctionId,
    pub definition: DefId,
    pub substitution: Substitution,
    pub name: String,
    pub type_parameters: Vec<GenericParamId>,
    pub parameters: Vec<FunctionParameter>,
    pub result: BoundaryType,
    pub compile_route: CompileRoute,
    pub call_type: CallType,
    pub body: Option<Block>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestDeclaration {
    pub name: String,
    pub modes: Vec<TestMode>,
    pub function: FunctionId,
    pub expectations: Vec<TestExpectation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestMode {
    Property,
    Benchmark,
    Chaos,
    Profile,
    Compiler,
    Integration,
}

impl TestMode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Property => "property",
            Self::Benchmark => "bench",
            Self::Chaos => "chaos",
            Self::Profile => "profile",
            Self::Compiler => "compiler",
            Self::Integration => "integ",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestExpectation {
    Contains { stream: TestStream, value: String },
    Excludes { stream: TestStream, value: String },
    Equals { stream: TestStream, value: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDeclaration {
    pub id: TypeId,
    pub name: String,
    pub interface: Option<InterfaceId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassFieldDeclaration {
    pub name: String,
    pub ty: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDeclaration {
    pub id: TypeId,
    pub name: String,
    pub fields: Vec<ClassFieldDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitMethodDeclaration {
    pub name: String,
    pub parameters: Vec<TraitType>,
    pub result: TraitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitType {
    SelfType,
    Concrete(TypeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub definition: DefId,
    pub name: String,
    pub methods: Vec<TraitMethodDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Program {
    pub modules: Vec<Module>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    /// Binding metadata is stored once; ordered execution sections refer to it
    /// by identity.
    pub bindings: Vec<Binding>,
    pub initializer: Block,
    pub functions: Vec<FunctionDeclaration>,
    pub entry: Option<FunctionId>,
    pub tests: Vec<TestDeclaration>,
    pub types: Vec<TypeDeclaration>,
    pub classes: Vec<ClassDeclaration>,
    /// Compile-time capability contracts. These survive lowering so every
    /// stage can validate its input, then are erased before MLIR emission.
    pub traits: Vec<TraitDeclaration>,
}
