#![forbid(unsafe_code)]

mod compile;
pub mod coverage;
mod error;
pub mod mutation;
mod runtime_asset;
mod test;
mod xla;

pub use compile::{
    check_path, compile_native, compile_native_with_options, compile_path, compile_source,
    inspect_toolchain, Compilation,
};
pub use error::CompileError;
pub use test::{compile_native_tests, native_test_compilation, native_test_count};
pub use xla::{collect_xla_kernels, XlaExecutionContext, XlaKernelArtifact};
