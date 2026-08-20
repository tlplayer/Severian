#![forbid(unsafe_code)]

mod context;
mod ids;
mod literal;
mod operator;
pub mod types;

pub use context::{UniversalContext, UniversalError};
pub use ids::{DeclarationId, PrimitiveId, TypeId};
pub use literal::{LiteralKind, LiteralValue};
pub use operator::{BinaryOperator, OperatorSignature, TypeConstraint, TypePattern, UnaryOperator};
pub use types::{
    FloatFormat, IntegerWidth, PrimitiveCategory, PrimitiveDefinition, PrimitiveRepresentation,
    ResolvedBinary, ResolvedUnary, TypeContext, TypeDefinition, TypeDefinitionKind, TypeError,
};
