#![forbid(unsafe_code)]

mod database;
mod model_graph;
mod tensor;

pub mod cpu;
pub mod gpu;
pub mod target;

pub use cpu::{CpuFeature, CpuTarget};
pub use gpu::{GpuDevice, GpuTarget, GpuVendor};
pub use target::{
    Architecture, Backend, OperatingSystem, Target, TargetError, TargetKind,
};

/// Existing platform runtime source surface.
pub fn database_source() -> &'static str {
    database::source()
}

/// Existing platform runtime source surface.
pub fn model_graph_source(rocm: bool) -> String {
    model_graph::source(rocm)
}

/// Existing platform runtime source surface.
pub fn tensor_source(
    relu: bool,
    add: bool,
    matmul: bool,
    transpose: bool,
    scale: bool,
    softmax_rows: bool,
    layer_norm: bool,
    relu_backward: bool,
    softmax_backward: bool,
    layer_norm_backward: bool,
    autodiff: bool,
    rocm: bool,
) -> String {
    tensor::source(
        relu,
        add,
        matmul,
        transpose,
        scale,
        softmax_rows,
        layer_norm,
        relu_backward,
        softmax_backward,
        layer_norm_backward,
        autodiff,
        rocm,
    )
}

/// Resolves a target string such as `native`, `xla`, `nvidia`, `amd`,
/// `cuda:sm_90`, or `rocm:gfx1100`.
pub fn resolve_target(specification: &str) -> Result<Target, TargetError> {
    Target::parse(specification)
}

/// Returns the native host target with detected CPU features.
pub fn native_target() -> Target {
    Target::native()
}
