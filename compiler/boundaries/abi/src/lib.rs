//! Severian ABI model.
//!
//! This crate describes how already-resolved values cross binary boundaries.
//! It deliberately does not know about FFI providers, packages, Tensor, Data,
//! XLA, Python, networking, or any other library/runtime domain.
//!
//! Generic ABI schemas are allowed here. Concrete layout/code generation is not:
//! every schema must be instantiated to `AbiType` / `AbiSignature` first.

pub mod model;
pub mod registry;
pub mod validate;

pub use model::*;
pub use registry::*;
pub use validate::*;
