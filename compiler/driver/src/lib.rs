#![forbid(unsafe_code)]

mod compile;
mod error;
mod test;

pub use compile::{
    check_path, compile_native, compile_path, compile_source, inspect_toolchain, Compilation,
};
pub use error::CompileError;
pub use test::{compile_native_tests, native_test_compilation};
