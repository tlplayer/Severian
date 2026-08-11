//! GPU lowering configuration.
//!
//! This module keeps target choice separate from HIR lowering. Dispatch/tiling
//! passes decide what should run on a GPU; this module describes how GPU MLIR
//! should be lowered for NVIDIA, AMD, or portable SPIR-V targets.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Nvidia,
    Amd,
    Spirv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self { x: 256, y: 1, z: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLoweringOptions {
    pub backend: GpuBackend,
    pub architecture: Option<String>,
    pub workgroup: WorkgroupSize,
    pub use_bare_ptr_call_conv: bool,
}

impl GpuLoweringOptions {
    pub fn nvidia(architecture: impl Into<String>) -> Self {
        Self {
            backend: GpuBackend::Nvidia,
            architecture: Some(architecture.into()),
            workgroup: WorkgroupSize::default(),
            use_bare_ptr_call_conv: true,
        }
    }

    pub fn amd(architecture: impl Into<String>) -> Self {
        Self {
            backend: GpuBackend::Amd,
            architecture: Some(architecture.into()),
            workgroup: WorkgroupSize::default(),
            use_bare_ptr_call_conv: true,
        }
    }

    pub fn spirv() -> Self {
        Self {
            backend: GpuBackend::Spirv,
            architecture: None,
            workgroup: WorkgroupSize::default(),
            use_bare_ptr_call_conv: false,
        }
    }
}

pub fn pass_pipeline(options: &GpuLoweringOptions) -> String {
    let mut passes = vec![
        "canonicalize".to_string(),
        "cse".to_string(),
        "one-shot-bufferize{bufferize-function-boundaries}".to_string(),
        "canonicalize".to_string(),
        "convert-linalg-to-parallel-loops".to_string(),
        "gpu-map-parallel-loops".to_string(),
        "convert-parallel-loops-to-gpu".to_string(),
        "canonicalize".to_string(),
        "convert-scf-to-cf".to_string(),
    ];

    match options.backend {
        GpuBackend::Nvidia => {
            passes.push("gpu-kernel-outlining".into());
            passes.push("convert-gpu-to-nvvm".into());
            passes.push("reconcile-unrealized-casts".into());
        }
        GpuBackend::Amd => {
            passes.push("gpu-kernel-outlining".into());
            passes.push("convert-gpu-to-rocdl".into());
            passes.push("reconcile-unrealized-casts".into());
        }
        GpuBackend::Spirv => {
            passes.push("gpu-kernel-outlining".into());
            passes.push("convert-gpu-to-spirv".into());
            passes.push("reconcile-unrealized-casts".into());
        }
    }

    format!("builtin.module({})", passes.join(","))
}

pub fn target_attribute(options: &GpuLoweringOptions) -> String {
    match options.backend {
        GpuBackend::Nvidia => {
            let chip = options.architecture.as_deref().unwrap_or("sm_80");
            format!("#nvvm.target<chip = \"{chip}\", features = \"+ptx80\", O = 2>")
        }

        GpuBackend::Amd => {
            let chip = options.architecture.as_deref().unwrap_or("gfx1100");
            format!("#rocdl.target<chip = \"{chip}\", O = 2>")
        }

        GpuBackend::Spirv => {
            "#spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>".into()
        }
    }
}

pub fn workgroup_attribute(size: WorkgroupSize) -> String {
    format!("workgroup_size = [{}, {}, {}]", size.x, size.y, size.z)
}

pub fn is_accelerator_backend(backend: GpuBackend) -> bool {
    matches!(
        backend,
        GpuBackend::Nvidia | GpuBackend::Amd | GpuBackend::Spirv
    )
}
