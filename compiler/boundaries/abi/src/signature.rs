use crate::{AbiId, AbiType, AbiTypeExpr};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterMode {
    In,
    Out,
    InOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ownership {
    Copy,
    Borrowed,
    Owned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lifetime {
    /// Valid only during the call.
    Call,
    /// Valid for the process/program lifetime.
    Static,
    /// Tied to parameter N of the same signature.
    Parameter(u16),
    /// Ownership/lifetime is returned to the caller.
    Return,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiValue {
    pub ty: AbiType,
    pub ownership: Ownership,
    pub lifetime: Lifetime,
}

impl AbiValue {
    pub fn copy(ty: AbiType) -> Self {
        Self { ty, ownership: Ownership::Copy, lifetime: Lifetime::Call }
    }

    pub fn borrowed(ty: AbiType) -> Self {
        Self { ty, ownership: Ownership::Borrowed, lifetime: Lifetime::Call }
    }

    pub fn owned(ty: AbiType) -> Self {
        Self { ty, ownership: Ownership::Owned, lifetime: Lifetime::Return }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiParameter {
    pub name: String,
    pub mode: ParameterMode,
    pub value: AbiValue,
}

impl AbiParameter {
    pub fn input(name: impl Into<String>, value: AbiValue) -> Self {
        Self { name: name.into(), mode: ParameterMode::In, value }
    }

    pub fn output(name: impl Into<String>, value: AbiValue) -> Self {
        Self { name: name.into(), mode: ParameterMode::Out, value }
    }

    pub fn inout(name: impl Into<String>, value: AbiValue) -> Self {
        Self { name: name.into(), mode: ParameterMode::InOut, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiSignature {
    pub abi: AbiId,
    pub parameters: Vec<AbiParameter>,
    pub returns: AbiValue,
    pub variadic: bool,
}

impl AbiSignature {
    pub fn new(abi: AbiId, parameters: Vec<AbiParameter>, returns: AbiValue) -> Self {
        Self { abi, parameters, returns, variadic: false }
    }
}

// Generic signature form. It is instantiated before validation/layout/lowering.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiValueExpr {
    pub ty: AbiTypeExpr,
    pub ownership: Ownership,
    pub lifetime: Lifetime,
}

impl AbiValueExpr {
    pub fn copy(ty: AbiTypeExpr) -> Self {
        Self { ty, ownership: Ownership::Copy, lifetime: Lifetime::Call }
    }

    pub fn borrowed(ty: AbiTypeExpr) -> Self {
        Self { ty, ownership: Ownership::Borrowed, lifetime: Lifetime::Call }
    }

    pub fn owned(ty: AbiTypeExpr) -> Self {
        Self { ty, ownership: Ownership::Owned, lifetime: Lifetime::Return }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiParameterExpr {
    pub name: String,
    pub mode: ParameterMode,
    pub value: AbiValueExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiSignatureExpr {
    pub abi: AbiId,
    pub parameters: Vec<AbiParameterExpr>,
    pub returns: AbiValueExpr,
    pub variadic: bool,
}
