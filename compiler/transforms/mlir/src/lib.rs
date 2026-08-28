#![deny(unsafe_op_in_unsafe_fn)]

mod emit;
mod ffi;
pub mod structured;
mod verify;

pub use emit::{render, MlirArtifact, MlirError};
pub use severian_lir::{
    LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape,
    LoweredType,
};
pub use verify::{
    compose, compose_gpu_launchers, verify_artifact, GpuLaunchArtifact, VerifiedMlirArtifact,
};

/// Canonical MLIR spelling for a lowered Severian type. Custom compilers use
/// the same scalar/tensor mapping as ordinary lowering.
pub fn type_spelling(ty: &LoweredType) -> Result<String, MlirError> {
    emit::mlir_type(ty)
}
