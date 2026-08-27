#![deny(unsafe_op_in_unsafe_fn)]

mod emit;
mod ffi;
mod verify;

pub use emit::{render, MlirArtifact, MlirError};
pub use severian_lir::{
    LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape,
    LoweredType,
};
pub use verify::{compose, verify_artifact, VerifiedMlirArtifact};

/// Canonical MLIR spelling for a lowered Severian type. Custom compilers use
/// the same scalar/tensor mapping as ordinary lowering.
pub fn type_spelling(ty: &LoweredType) -> Result<String, MlirError> {
    emit::mlir_type(ty)
}
