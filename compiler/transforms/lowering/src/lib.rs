#![forbid(unsafe_code)]

pub mod coverage;
pub mod debug;
pub mod gpu;
pub mod kernel;
pub mod llvm;
pub mod location;
pub mod runtime;
pub mod stablehlo;
pub mod tensor;

mod core;
pub use core::{lower, native_bridge_source, rocm_bridge_source};
