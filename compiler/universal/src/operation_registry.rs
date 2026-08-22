use crate::{TyId, TypeContext};
use std::collections::BTreeMap;
use std::sync::Arc;

const fn hash(value: &str) -> u128 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DialectId(pub u128);

impl DialectId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationId(pub u128);

impl OperationId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId {
    pub dialect: DialectId,
    pub operation: OperationId,
}

impl OpId {
    pub const fn named(dialect: &str, operation: &str) -> Self {
        Self {
            dialect: DialectId::from_name(dialect),
            operation: OperationId::from_name(operation),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttributeId(pub u128);

impl AttributeId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrValue {
    Integer(i128),
    Boolean(bool),
    String(String),
    Type(TyId),
    Types(Vec<TyId>),
}

pub type Attrs = BTreeMap<AttributeId, AttrValue>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectSet(pub u16);

impl EffectSet {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const ALLOCATE: Self = Self(1 << 2);
    pub const FREE: Self = Self(1 << 3);
    pub const THROW: Self = Self(1 << 4);
    pub const IO: Self = Self(1 << 5);

    pub const fn contains(self, effect: Self) -> bool {
        self.0 & effect.0 == effect.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringCapability {
    Standard,
    Compiler(crate::CompilerId),
    Backend(String),
    Runtime(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredOperation {
    pub id: OpId,
    pub operands: Vec<TyId>,
    pub results: Vec<TyId>,
    pub attributes: Attrs,
}

pub struct IrContext<'a> {
    pub types: &'a TypeContext,
    pub operations: &'a OperationRegistry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationDiagnostic {
    pub operation: OpId,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalRewrite {
    pub replacement: OpId,
    pub attributes: Attrs,
}

pub trait OperationInterface: Send + Sync {
    fn infer_types(
        &self,
        operands: &[TyId],
        attributes: &Attrs,
    ) -> Result<Vec<TyId>, OperationDiagnostic>;

    fn verify(
        &self,
        operation: &RegisteredOperation,
        context: &IrContext<'_>,
    ) -> Result<(), OperationDiagnostic>;

    fn effects(&self, operation: &RegisteredOperation) -> EffectSet;

    fn canonicalize(&self, operation: &RegisteredOperation) -> Option<CanonicalRewrite>;

    fn lowering_capabilities(&self) -> &[LoweringCapability];
}

#[derive(Default)]
pub struct OperationRegistry {
    interfaces: BTreeMap<OpId, Arc<dyn OperationInterface>>,
}

impl OperationRegistry {
    pub fn register(
        &mut self,
        id: OpId,
        interface: impl OperationInterface + 'static,
    ) -> Result<(), OperationDiagnostic> {
        if self.interfaces.insert(id, Arc::new(interface)).is_some() {
            return Err(OperationDiagnostic {
                operation: id,
                message: "operation interface is already registered".into(),
            });
        }
        Ok(())
    }

    pub fn interface(&self, id: OpId) -> Option<&dyn OperationInterface> {
        self.interfaces.get(&id).map(Arc::as_ref)
    }
}
