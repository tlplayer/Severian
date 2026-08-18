//! Severian ABI model.
//!
//! This crate describes how already-resolved values cross binary boundaries.
//! It deliberately does not know about FFI providers, packages, Tensor, Data,
//! XLA, Python, networking, or any other library/runtime domain.
//!
//! Generic ABI schemas are allowed here. Concrete layout/code generation is not:
//! every schema must be instantiated to `AbiType` / `AbiSignature` first.

pub mod convention;
pub mod id;
pub mod instantiate;
pub mod layout;
pub mod registry;
pub mod schema;
pub mod signature;
pub mod types;
pub mod validate;

pub use convention::*;
pub use id::*;
pub use instantiate::*;
pub use layout::*;
pub use registry::*;
pub use schema::*;
pub use signature::*;
pub use types::*;
pub use validate::*;
