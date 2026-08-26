#![forbid(unsafe_code)]

mod context;
mod ids;
mod literal;
mod operation_registry;
mod operator;
pub mod primitive;
mod type_system;
pub mod types;

pub use context::{UniversalContext, UniversalError};
pub use ids::{
    CompilerId, DeclarationId, DefId, GenericParamId, InferVarId, InstanceId, PrimitiveId,
    RegionId, TyId, TypeId,
};
pub use literal::{LiteralKind, LiteralValue};
pub use operation_registry::{
    AttrValue, AttributeId, Attrs, BackendId, CanonicalRewrite, DialectId, EffectSet, IrContext,
    LoweringCapability, OpId, OperationDiagnostic, OperationId, OperationInterface,
    OperationRegistry, ProviderId, RegisteredOperation, RuntimeId,
};
pub use operator::{BinaryOperator, OperatorSignature, TypeConstraint, TypePattern, UnaryOperator};
pub use primitive::{
    install_primitives, FloatFormat, IntegerWidth, PrimitiveCategory, PrimitiveDefinition,
    PrimitiveRepresentation, PrimitiveSpec, PRIMITIVES,
};
pub use type_system::{
    Constraint, ImplDefinition, ImplId, ImplSelection, ImplTable, InferenceContext, Signature,
    Substitution, TraitRef, TyInterner, TypeKind, UnifyError,
};
pub use types::{
    ResolvedBinary, ResolvedUnary, TypeContext, TypeContextBuilder, TypeDefinition,
    TypeDefinitionKind, TypeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileRoute {
    Standard,
    Compiler(CompilerId),
}

/// Stable identity for a source-level `*[T]`. Raw pointers are structural
/// native values, not nominal classes and not entries in the primitive table.
pub const fn raw_pointer_type_id(element: TypeId) -> TypeId {
    let hash = (0x811c_9dc5u32 ^ element.0).wrapping_mul(0x0100_0193);
    TypeId(0x0800_0000 | (hash & 0x03ff_ffff))
}

pub const fn is_raw_pointer_type(ty: TypeId) -> bool {
    ty.0 & 0xfc00_0000 == 0x0800_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_pointer_ids_are_stable_distinct_and_recognizable() {
        let first = raw_pointer_type_id(TypeId(1));
        let same = raw_pointer_type_id(TypeId(1));
        let second = raw_pointer_type_id(TypeId(2));
        assert_eq!(first, same);
        assert_ne!(first, second);
        assert!(is_raw_pointer_type(first));
        assert!(!is_raw_pointer_type(TypeId(1)));
    }
}
