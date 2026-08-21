#![forbid(unsafe_code)]

#[path = "model/expression/mod.rs"]
mod expression;
#[path = "model/statement/mod.rs"]
mod statement;

pub use expression::{Expression, ExpressionKind};
pub use severian_universal::{CompileRoute, CompilerId, TypeId};
pub use statement::{Binding, Block, Statement};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeclaredTypeId(pub u32);

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
pub enum SemanticType {
    Universal(TypeId),
    Declared(DeclaredTypeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryModifier {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryType {
    pub ty: SemanticType,
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
    pub name: String,
    pub parameters: Vec<FunctionParameter>,
    pub result: BoundaryType,
    pub compile_route: CompileRoute,
    pub call_type: CallType,
    pub body: Option<Block>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDeclaration {
    pub id: DeclaredTypeId,
    pub name: String,
    pub interface: Option<InterfaceId>,
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
    pub types: Vec<TypeDeclaration>,
}
