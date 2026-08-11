use crate::{
    linker::{link_native_executable, NativeLinkOptions},
    llvm::{lower_module_to_llvm_ir, LlvmLoweringOptions},
    toolchain::TemporaryFiles,
    BackendError,
};
use severian_hir::Program;
use severian_mlir::Module;
use std::{ffi::OsString, path::Path};

/// Host-native compilation.
///
/// This is the modular equivalent of the existing monolithic
/// `severian_backend::compile_native`.
pub fn compile_native(
    program: &Program,
    module: &Module,
    output: &Path,
) -> Result<(), BackendError> {
    let temporary = TemporaryFiles::new("severian-native");
    let source_mlir = temporary.path("source.mlir");
    let lowered_mlir = temporary.path("llvm.mlir");
    let llvm_ir = temporary.path("module.ll");
    let bridge_source = temporary.path("runtime.c");

    std::fs::write(&source_mlir, module.as_str())?;

    lower_module_to_llvm_ir(
        &source_mlir,
        &lowered_mlir,
        &llvm_ir,
        &LlvmLoweringOptions::native(),
    )?;

    let bridge = severian_lowering::native_bridge_source(program);
    let bridge_path = if bridge.is_empty() {
        None
    } else {
        std::fs::write(&bridge_source, bridge)?;
        Some(bridge_source.as_path())
    };

    let uses_database = program.functions.iter().any(|function| {
        function
            .native_symbol
            .as_deref()
            .is_some_and(|symbol| symbol.starts_with("__sev_database_"))
    });

    link_native_executable(
        &llvm_ir,
        bridge_path,
        output,
        &NativeLinkOptions {
            sqlite: uses_database,
            pthread: bridge_path.is_some(),
            math: true,
            optimization: 3,
            additional_arguments: vec![
                OsString::from("-ffunction-sections"),
                OsString::from("-fdata-sections"),
                OsString::from("-Wl,--gc-sections"),
            ],
            ..NativeLinkOptions::default()
        },
    )
}
