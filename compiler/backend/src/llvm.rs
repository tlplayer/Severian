use crate::{
    toolchain::{find_required_tool, run_tool, Tool},
    BackendError,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LlvmLoweringOptions {
    pub linalg_to_parallel: bool,
    pub lower_vectors: bool,
    pub index_bitwidth: Option<u8>,
    pub additional_passes: Vec<String>,
}

impl LlvmLoweringOptions {
    pub fn native() -> Self {
        Self {
            linalg_to_parallel: false,
            lower_vectors: true,
            index_bitwidth: None,
            additional_passes: Vec::new(),
        }
    }

    pub fn host_after_gpu() -> Self {
        Self {
            linalg_to_parallel: false,
            lower_vectors: true,
            index_bitwidth: None,
            additional_passes: Vec::new(),
        }
    }
}

impl Default for LlvmLoweringOptions {
    fn default() -> Self {
        Self::native()
    }
}

pub fn llvm_lowering_passes(options: &LlvmLoweringOptions) -> Vec<String> {
    let mut passes = Vec::new();

    if options.linalg_to_parallel {
        passes.push("--convert-linalg-to-parallel-loops".into());
    } else {
        passes.push("--convert-linalg-to-loops".into());
    }

    passes.extend([
        "--lower-affine".into(),
        "--convert-scf-to-cf".into(),
        "--convert-index-to-llvm".into(),
        "--convert-math-to-llvm".into(),
        "--convert-arith-to-llvm".into(),
    ]);

    if options.lower_vectors {
        passes.push("--convert-vector-to-llvm".into());
    }

    passes.push(match options.index_bitwidth {
        Some(bits) => format!("--finalize-memref-to-llvm=index-bitwidth={bits}"),
        None => "--finalize-memref-to-llvm".into(),
    });

    passes.extend([
        "--convert-func-to-llvm".into(),
        "--convert-cf-to-llvm".into(),
        "--reconcile-unrealized-casts".into(),
    ]);

    passes.extend(options.additional_passes.iter().cloned());
    passes
}

pub fn lower_module_to_llvm_ir(
    source_mlir: &Path,
    lowered_mlir: &Path,
    llvm_ir: &Path,
    options: &LlvmLoweringOptions,
) -> Result<(), BackendError> {
    let mlir_opt = find_required_tool(Tool::MlirOpt)?;
    let translate = find_required_tool(Tool::MlirTranslate)?;

    let mut arguments = vec![source_mlir.as_os_str().to_owned()];
    for pass in llvm_lowering_passes(options) {
        arguments.push(pass.into());
    }
    arguments.push("-o".into());
    arguments.push(lowered_mlir.as_os_str().to_owned());

    run_tool(&mlir_opt, &arguments)?;

    run_tool(
        &translate,
        &[
            "--mlir-to-llvmir".into(),
            lowered_mlir.as_os_str().to_owned(),
            "-o".into(),
            llvm_ir.as_os_str().to_owned(),
        ],
    )
}

pub fn translate_llvm_dialect_to_ir(
    lowered_mlir: &Path,
    llvm_ir: &Path,
) -> Result<(), BackendError> {
    let translate = find_required_tool(Tool::MlirTranslate)?;
    run_tool(
        &translate,
        &[
            "--mlir-to-llvmir".into(),
            lowered_mlir.as_os_str().to_owned(),
            "-o".into(),
            llvm_ir.as_os_str().to_owned(),
        ],
    )
}

pub fn optimize_llvm_ir(
    input: &Path,
    output: &Path,
    level: u8,
) -> Result<(), BackendError> {
    let optimizer = find_required_tool(Tool::Opt)?;
    run_tool(
        &optimizer,
        &[
            format!("-O{}", level.min(3)).into(),
            input.as_os_str().to_owned(),
            "-o".into(),
            output.as_os_str().to_owned(),
        ],
    )
}
