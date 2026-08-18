use crate::{CompileTypeId, Constraint, ExternalId, GenericParameter, InterfaceType, IntrinsicId};

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionInterface {
    pub parameters: Vec<Parameter>,
    pub returns: InterfaceType,
    pub generics: Vec<GenericParameter>,
    pub constraints: Vec<Constraint>,
    pub implementation: FunctionImplementation,
    pub safety: Safety,

    /// Compiler domain that owns lowering of this operation.
    ///
    /// This is explicit on functions because domain operations can return core
    /// values (for example a tensor reduction returning a scalar). Methods may
    /// inherit the enclosing class compile type when the interface is emitted.
    pub compile_type: Option<CompileTypeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub ty: InterfaceType,
    pub mode: PassingMode,
    pub default: Option<ConstantValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassingMode {
    Value,
    Borrowed,
    MutableBorrowed,
    Owned,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionImplementation {
    Defined,
    Required,
    External(ExternalId),
    Intrinsic(IntrinsicId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Safety {
    Safe,
    Unsafe,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstantValue {
    Unit,
    None,
    Bool(bool),
    Int(i128),
    UInt(u128),
    Float(f64),
    String(String),
    Tuple(Vec<ConstantValue>),
}
