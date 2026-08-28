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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId(pub u128);

impl BackendId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeId(pub u128);

impl RuntimeId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProviderId(pub u128);

impl ProviderId {
    pub const fn from_name(name: &str) -> Self {
        Self(hash(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrValue {
    Integer(i128),
    Integers(Vec<i128>),
    Boolean(bool),
    String(String),
    Type(TyId),
    Types(Vec<TyId>),
    TensorShape(crate::TensorShape),
    Compiler(crate::CompilerId),
}

pub const COMPILE_TYPE_ATTRIBUTE: AttributeId = AttributeId::from_name("compile.type");
pub const COMPILE_TARGETS_ATTRIBUTE: AttributeId = AttributeId::from_name("compile.targets");
pub const COMPILED_ARTIFACT_ATTRIBUTE: AttributeId = AttributeId::from_name("compile.artifact");

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
    Backend(BackendId),
    Runtime(RuntimeId),
    Provider(ProviderId),
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

#[derive(Clone, Default)]
pub struct OperationRegistry {
    interfaces: BTreeMap<OpId, Arc<dyn OperationInterface>>,
}

impl std::fmt::Debug for OperationRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OperationRegistry")
            .field("interfaces", &self.interfaces.len())
            .finish()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiteralKind, TypeContextBuilder, TypeId};

    #[derive(Clone)]
    struct TestInterface {
        result: TypeId,
        capabilities: Vec<LoweringCapability>,
    }

    impl OperationInterface for TestInterface {
        fn infer_types(
            &self,
            operands: &[TyId],
            _attributes: &Attrs,
        ) -> Result<Vec<TyId>, OperationDiagnostic> {
            if operands.is_empty() {
                Err(OperationDiagnostic {
                    operation: OpId::named("test", "add"),
                    message: "an operand is required".into(),
                })
            } else {
                Ok(vec![self.result])
            }
        }

        fn verify(
            &self,
            operation: &RegisteredOperation,
            _context: &IrContext<'_>,
        ) -> Result<(), OperationDiagnostic> {
            if operation.results == [self.result] {
                Ok(())
            } else {
                Err(OperationDiagnostic {
                    operation: operation.id,
                    message: "unexpected result type".into(),
                })
            }
        }

        fn effects(&self, _operation: &RegisteredOperation) -> EffectSet {
            EffectSet::READ.union(EffectSet::WRITE)
        }

        fn canonicalize(&self, operation: &RegisteredOperation) -> Option<CanonicalRewrite> {
            operation.attributes.is_empty().then(|| CanonicalRewrite {
                replacement: OpId::named("test", "canonical"),
                attributes: Attrs::new(),
            })
        }

        fn lowering_capabilities(&self) -> &[LoweringCapability] {
            &self.capabilities
        }
    }

    #[test]
    fn stable_operation_identifiers_and_effect_sets_are_composable() {
        const OPERATION: OpId = OpId::named("arith", "add");
        assert_eq!(OPERATION.dialect, DialectId::from_name("arith"));
        assert_eq!(OPERATION.operation, OperationId::from_name("add"));
        assert_ne!(OPERATION.operation, OperationId::from_name("subtract"));
        assert_ne!(
            AttributeId::from_name("type"),
            AttributeId::from_name("value")
        );
        assert_ne!(BackendId::from_name("native"), BackendId::from_name("wasm"));
        assert_ne!(RuntimeId::from_name("host"), RuntimeId::from_name("device"));
        assert_ne!(ProviderId::from_name("cpu"), ProviderId::from_name("gpu"));

        let effects = EffectSet::NONE
            .union(EffectSet::READ)
            .union(EffectSet::WRITE)
            .union(EffectSet::ALLOCATE)
            .union(EffectSet::FREE)
            .union(EffectSet::THROW)
            .union(EffectSet::IO);
        for effect in [
            EffectSet::READ,
            EffectSet::WRITE,
            EffectSet::ALLOCATE,
            EffectSet::FREE,
            EffectSet::THROW,
            EffectSet::IO,
        ] {
            assert!(effects.contains(effect));
        }
        assert!(!EffectSet::READ.contains(EffectSet::WRITE));
    }

    #[test]
    fn registry_rejects_duplicates_and_exposes_interface_behavior() {
        let id = OpId::named("test", "add");
        let capabilities = vec![
            LoweringCapability::Standard,
            LoweringCapability::Backend(BackendId::from_name("native")),
            LoweringCapability::Runtime(RuntimeId::from_name("host")),
            LoweringCapability::Provider(ProviderId::from_name("cpu")),
        ];
        let interface = TestInterface {
            result: TypeId(7),
            capabilities: capabilities.clone(),
        };
        let mut registry = OperationRegistry::default();
        registry.register(id, interface.clone()).unwrap();
        let duplicate = registry.register(id, interface).unwrap_err();
        assert_eq!(duplicate.operation, id);
        assert_eq!(
            duplicate.message,
            "operation interface is already registered"
        );
        assert!(registry
            .interface(OpId::named("missing", "operation"))
            .is_none());
        assert_eq!(
            format!("{registry:?}"),
            "OperationRegistry { interfaces: 1 }"
        );

        let interface = registry.interface(id).unwrap();
        assert_eq!(
            interface.infer_types(&[TypeId(1)], &Attrs::new()).unwrap(),
            [TypeId(7)]
        );
        assert!(interface.infer_types(&[], &Attrs::new()).is_err());
        assert_eq!(interface.lowering_capabilities(), capabilities);

        let mut types = TypeContextBuilder::new();
        types.register_declaration("test.Value", "Value").unwrap();
        let types = types.build();
        let context = IrContext {
            types: &types,
            operations: &registry,
        };
        let operation = RegisteredOperation {
            id,
            operands: vec![TypeId(1)],
            results: vec![TypeId(7)],
            attributes: Attrs::new(),
        };
        interface.verify(&operation, &context).unwrap();
        assert!(interface.effects(&operation).contains(EffectSet::READ));
        assert_eq!(
            interface.canonicalize(&operation).unwrap().replacement,
            OpId::named("test", "canonical")
        );

        let mut invalid = operation.clone();
        invalid.results = vec![TypeId(8)];
        assert!(interface.verify(&invalid, &context).is_err());
        invalid.attributes.insert(
            AttributeId::from_name("literal"),
            AttrValue::String(format!("{:?}", LiteralKind::Integer)),
        );
        assert_eq!(interface.canonicalize(&invalid), None);
    }
}
