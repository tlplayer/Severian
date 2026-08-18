//! MLIR -> LLVM lowering pipeline description.
//!
//! The lowering crate currently emits textual MLIR, so this module describes
//! the canonical pass pipeline instead of binding directly to MLIR C++ APIs.
//! The driver can feed `pass_pipeline()` to mlir-opt or reproduce these passes
//! through the MLIR C API later.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

#[derive(Debug, Clone)]
pub struct LlvmLoweringOptions {
    pub optimization: OptimizationLevel,
    pub enable_vectorization: bool,
    pub enable_parallel_loops: bool,
    pub index_bitwidth: u8,
}

impl Default for LlvmLoweringOptions {
    fn default() -> Self {
        Self {
            optimization: OptimizationLevel::O2,
            enable_vectorization: true,
            enable_parallel_loops: false,
            index_bitwidth: 64,
        }
    }
}

pub fn pass_pipeline(options: &LlvmLoweringOptions) -> String {
    let mut passes = Vec::<String>::new();

    passes.push("canonicalize".into());
    passes.push("cse".into());

    // Tensor values must become buffers before conversion to LLVM.
    passes.push("one-shot-bufferize{bufferize-function-boundaries}".into());
    passes.push("canonicalize".into());

    if options.enable_vectorization {
        passes.push("convert-linalg-to-loops".into());
        passes.push("canonicalize".into());
        passes.push("convert-vector-to-scf".into());
    } else if options.enable_parallel_loops {
        passes.push("convert-linalg-to-parallel-loops".into());
    } else {
        passes.push("convert-linalg-to-loops".into());
    }

    passes.extend([
        "lower-affine".into(),
        "convert-scf-to-cf".into(),
        "convert-math-to-llvm".into(),
        "convert-arith-to-llvm".into(),
        format!(
            "finalize-memref-to-llvm{{index-bitwidth={}}}",
            options.index_bitwidth
        ),
        "convert-func-to-llvm".into(),
        "convert-cf-to-llvm".into(),
        "reconcile-unrealized-casts".into(),
    ]);

    format!("builtin.module({})", passes.join(","))
}

pub fn translation_command(input: &str, output: &str) -> String {
    format!("mlir-translate --mlir-to-llvmir {input} -o {output}")
}

pub fn optimizer_command(input: &str, output: &str, options: &LlvmLoweringOptions) -> String {
    let level = match options.optimization {
        OptimizationLevel::O0 => "-O0",
        OptimizationLevel::O1 => "-O1",
        OptimizationLevel::O2 => "-O2",
        OptimizationLevel::O3 => "-O3",
    };

    format!("opt {level} {input} -o {output}")
}
