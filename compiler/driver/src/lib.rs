#![forbid(unsafe_code)]

pub mod artifact;
pub mod compile;
mod error;
mod frontend;
pub mod options;
pub mod pipeline;
pub mod target;
mod test;

pub use compile::{compile, CompileInput, CompileOutput, CompileRequest};
pub use error::CompileError;
pub use frontend::{
    check_path, compile_native, compile_path, compile_source, inspect_toolchain, Compilation,
};
pub use options::{CompileOptions as DriverCompileOptions, EmitKind, OptimizationLevel};
pub use pipeline::{PipelinePlan, PipelineStage};
pub use target::{BackendFamily, DriverTarget, TargetParseError};
pub use test::{compile_native_tests, native_test_compilation};
