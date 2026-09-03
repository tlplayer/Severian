use severian_abi::{AbiType, CallingConvention, Symbol};
use severian_universal::TypeId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lifetime {
    Call,
    Return,
    Static,
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ownership {
    Copy,
    Borrowed(Lifetime),
    Owned,
    Transferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterMode {
    In,
    Out,
    InOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForeignTypeRef {
    Severian(TypeId),
    External(String),
    Pointer { pointee: Box<Self>, mutable: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueContract {
    pub ty: ForeignTypeRef,
    pub ownership: Ownership,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignParameter {
    pub name: String,
    pub contract: ValueContract,
    pub mode: ParameterMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiSelection {
    C,
    System,
    Rust,
    Explicit(CallingConvention),
}

impl AbiSelection {
    pub const fn convention(self) -> CallingConvention {
        match self {
            Self::C => CallingConvention::C,
            Self::System => CallingConvention::System,
            Self::Rust => CallingConvention::Rust,
            Self::Explicit(convention) => convention,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignFunction {
    pub name: String,
    pub provider: Option<String>,
    pub symbol: Symbol,
    pub parameters: Vec<ForeignParameter>,
    pub result: ValueContract,
    pub abi: AbiSelection,
    pub variadic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignTypeDeclaration {
    pub name: String,
    pub representation: AbiType,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ForeignModule {
    pub types: Vec<ForeignTypeDeclaration>,
    pub functions: Vec<ForeignFunction>,
}

impl ForeignModule {
    pub fn type_declaration(&self, name: &str) -> Option<&ForeignTypeDeclaration> {
        self.types
            .iter()
            .find(|declaration| declaration.name == name)
    }
}
