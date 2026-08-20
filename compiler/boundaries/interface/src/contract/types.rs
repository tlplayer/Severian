use crate::{CapabilityId, GenericId, SymbolId, TypeId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterfaceType {
    Any,
    Unit,
    None,
    Bool,
    Int(IntType),
    Float(FloatType),
    String,
    Tuple(Vec<InterfaceType>),
    List(Box<InterfaceType>),
    Set(Box<InterfaceType>),
    Map(Box<InterfaceType>, Box<InterfaceType>),
    Named(TypeId),
    Generic(GenericId),
    Function(FunctionType),
    Union(Vec<InterfaceType>),
    Reference(ReferenceType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntType {
    pub signed: bool,
    pub bits: u16,
}

impl IntType {
    pub const fn signed(bits: u16) -> Self {
        Self { signed: true, bits }
    }

    pub const fn unsigned(bits: u16) -> Self {
        Self {
            signed: false,
            bits,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatType {
    pub bits: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionType {
    pub parameters: Vec<InterfaceType>,
    pub returns: Box<InterfaceType>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceType {
    pub target: Box<InterfaceType>,
    pub mutable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenericParameter {
    pub id: GenericId,
    pub name: String,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constraint {
    Implements {
        ty: InterfaceType,
        trait_id: TypeId,
    },
    SameType {
        left: InterfaceType,
        right: InterfaceType,
    },
    Capability(CapabilityId),
    Predicate(SymbolId),
}
