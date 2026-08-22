#![forbid(unsafe_code)]

mod context;
mod ids;
mod literal;
mod operation_registry;
mod operator;
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
pub use type_system::{
    Constraint, ImplDefinition, ImplId, ImplSelection, ImplTable, InferenceContext, Signature,
    Substitution, TraitRef, TyInterner, TyKind, UnifyError,
};
pub use types::{
    FloatFormat, IntegerWidth, PrimitiveCategory, PrimitiveDefinition, PrimitiveRepresentation,
    ResolvedBinary, ResolvedUnary, TypeContext, TypeContextBuilder, TypeDefinition,
    TypeDefinitionKind, TypeError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileRoute {
    Standard,
    Compiler(CompilerId),
}
