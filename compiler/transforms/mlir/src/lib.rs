#![deny(unsafe_op_in_unsafe_fn)]

mod emit;
mod ffi;
mod verify;

pub use emit::{render, MlirArtifact, MlirError};
pub use severian_lir::{LoweredFloatFormat, LoweredType};
pub use verify::{compose, verify_artifact, VerifiedMlirArtifact};
