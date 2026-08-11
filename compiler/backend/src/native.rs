use crate::{
    linker::{link_native_executable, NativeLinkOptions},
    llvm::{lower_module_to_llvm_ir, LlvmLoweringOptions},
    toolchain::TemporaryFiles,
    BackendError,
};
use severian_hir::Program;
use severian_mlir::Module;
use std::{ffi::OsString, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSanitizer {
    Address,
    Thread,
    Memory,
    Undefined,
}

#[derive(Debug, Clone, Default)]
pub struct NativeCompileOptions {
    pub sanitizers: Vec<NativeSanitizer>,
}

/// Host-native compilation.
///
/// This is the modular equivalent of the existing monolithic
/// `severian_backend::compile_native`.
pub fn compile_native(
    program: &Program,
    module: &Module,
    output: &Path,
    xla_runtime: Option<&Path>,
    options: &NativeCompileOptions,
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

    let bridge = severian_lowering::native_bridge_source(program)
        .map_err(|error| BackendError(std::io::Error::other(error)))?;
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
    let uses_xla = program.functions.iter().any(|function| {
        function
            .decorators
            .iter()
            .any(|decorator| decorator.package == "tensor")
    });
    let mut libraries = Vec::new();
    if uses_xla {
        let library = xla_runtime.ok_or_else(|| {
            BackendError(std::io::Error::other(
                "native program contains XLA tensor regions but no runtime archive was supplied",
            ))
        })?;
        libraries.push(library.to_path_buf());
    }

    let sanitizer_names = options
        .sanitizers
        .iter()
        .map(|sanitizer| match sanitizer {
            NativeSanitizer::Address => "address",
            NativeSanitizer::Thread => "thread",
            NativeSanitizer::Memory => "memory",
            NativeSanitizer::Undefined => "undefined",
        })
        .collect::<Vec<_>>();
    let mut additional_arguments = vec![
        OsString::from("-ffunction-sections"),
        OsString::from("-fdata-sections"),
        OsString::from("-Wl,--gc-sections"),
        OsString::from("-ldl"),
        OsString::from("-lrt"),
        OsString::from("-lutil"),
    ];
    if !sanitizer_names.is_empty() {
        additional_arguments.extend([
            format!("-fsanitize={}", sanitizer_names.join(",")).into(),
            OsString::from("-fno-omit-frame-pointer"),
            OsString::from("-g"),
        ]);
    }

    link_native_executable(
        &llvm_ir,
        bridge_path,
        output,
        &NativeLinkOptions {
            sqlite: uses_database,
            libraries,
            pthread: bridge_path.is_some(),
            math: true,
            optimization: if sanitizer_names.is_empty() { 3 } else { 1 },
            additional_arguments,
            ..NativeLinkOptions::default()
        },
    )
}
