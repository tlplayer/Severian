//! Backend-neutral tensor-kernel planning and specialized backend emission.
//!
//! The kernel IR describes semantics once. Backend policy then routes portable
//! graphs through StableHLO/XLA and recognized GPU kernels through native
//! Triton IR. Neither path depends on Python or a benchmark harness.

mod ir;
mod selection;
mod stablehlo;
mod triton;

pub use ir::{collect, find};
pub use selection::select_backend;
pub use stablehlo::emit_stablehlo;
pub use triton::{emit_triton_ir, TritonLaunch};

use severian_hir::{FunctionId, TensorElementType, TensorType};
use severian_platform::{resolve_target, GpuVendor, Target};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Auto,
    Xla,
    Triton,
    Llvm,
}

impl KernelBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Xla => "xla",
            Self::Triton => "triton",
            Self::Llvm => "llvm",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelTarget {
    Cpu,
    /// A deployment GPU whose vendor has not been fixed yet.
    Gpu,
    Nvidia {
        architecture: Option<String>,
    },
    Amd {
        architecture: Option<String>,
    },
    Tpu,
}

impl KernelTarget {
    pub fn parse(specification: &str) -> Result<Self, String> {
        match specification.trim().to_ascii_lowercase().as_str() {
            "gpu" => return Ok(Self::Gpu),
            "tpu" => return Ok(Self::Tpu),
            _ => {}
        }
        match resolve_target(specification).map_err(|error| error.to_string())? {
            Target::Cpu(_) => Ok(Self::Cpu),
            Target::Gpu(target) => match target.vendor {
                GpuVendor::Nvidia => Ok(Self::Nvidia {
                    architecture: target.architecture,
                }),
                GpuVendor::Amd => Ok(Self::Amd {
                    architecture: target.architecture,
                }),
                GpuVendor::Unknown => Ok(Self::Gpu),
            },
            Target::Xla { platform, .. } => {
                if platform.as_deref() == Some("tpu") {
                    Ok(Self::Tpu)
                } else {
                    Ok(Self::Gpu)
                }
            }
            Target::Spirv { .. } => Ok(Self::Gpu),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Gpu => "gpu".into(),
            Self::Nvidia { architecture } => architecture
                .as_deref()
                .map_or_else(|| "nvidia".into(), |arch| format!("cuda:{arch}")),
            Self::Amd { architecture } => architecture
                .as_deref()
                .map_or_else(|| "amd".into(), |arch| format!("rocm:{arch}")),
            Self::Tpu => "tpu".into(),
        }
    }

    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu | Self::Nvidia { .. } | Self::Amd { .. })
    }

    /// Returns whether automatic policy may select Triton for this target.
    pub fn supports_triton(&self) -> bool {
        match self {
            Self::Nvidia {
                architecture: Some(architecture),
            } => supported_nvidia_architecture(architecture),
            Self::Amd {
                architecture: Some(_),
            } => true,
            Self::Cpu | Self::Gpu | Self::Tpu => false,
            Self::Nvidia { architecture: None } | Self::Amd { architecture: None } => false,
        }
    }

    pub fn is_known_triton_incompatible(&self) -> bool {
        matches!(
            self,
            Self::Nvidia {
                architecture: Some(architecture)
            } if !supported_nvidia_architecture(architecture)
        )
    }
}

fn supported_nvidia_architecture(architecture: &str) -> bool {
    architecture
        .strip_prefix("sm_")
        .and_then(|value| value.trim_end_matches(['a', 'f']).parse::<u16>().ok())
        .is_some_and(|compute_capability| compute_capability >= 80)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOperation {
    ReductionSum { input: usize },
    ElementwiseRelu { input: usize },
}

impl KernelOperation {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ReductionSum { .. } => "reduction.sum",
            Self::ElementwiseRelu { .. } => "elementwise.relu",
        }
    }

    pub const fn input(&self) -> usize {
        match self {
            Self::ReductionSum { input } | Self::ElementwiseRelu { input } => *input,
        }
    }

    pub const fn supports_triton(&self) -> bool {
        matches!(
            self,
            Self::ReductionSum { .. } | Self::ElementwiseRelu { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIr {
    pub function: FunctionId,
    pub name: String,
    pub parameters: Vec<TensorType>,
    pub result: TensorType,
    pub operation: KernelOperation,
    pub policy: KernelBackend,
}

impl KernelIr {
    pub fn triton_support(&self) -> Result<(), String> {
        if !self.operation.supports_triton() {
            return Err(format!(
                "operation `{}` has no direct Triton lowering",
                self.operation.name()
            ));
        }
        let input = self
            .parameters
            .get(self.operation.input())
            .ok_or_else(|| format!("kernel input {} does not exist", self.operation.input()))?;
        match (self.operation, input.element) {
            (KernelOperation::ReductionSum { .. }, TensorElementType::F32) => Ok(()),
            (KernelOperation::ReductionSum { .. }, element) => Err(format!(
                "reduction.sum Triton lowering currently requires f32, found {}",
                tensor_element_name(element)
            )),
            (KernelOperation::ElementwiseRelu { .. }, element)
                if element.satisfies(severian_hir::TensorElementConstraint::Float) =>
            {
                Ok(())
            }
            (KernelOperation::ElementwiseRelu { .. }, element) => Err(format!(
                "elementwise.relu requires a floating-point tensor, found {}",
                tensor_element_name(element)
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSelection {
    pub requested: KernelBackend,
    pub selected: KernelBackend,
    pub target: KernelTarget,
    pub reason: String,
    pub fallback: Option<KernelBackend>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    NoKernels,
    UnknownEntry(String),
    AmbiguousEntries(Vec<String>),
    UnsupportedBackend {
        kernel: String,
        backend: KernelBackend,
        reason: String,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKernels => formatter.write_str(
                "no specialized tensor kernels were found; direct return expressions currently support reduction sum and ReLU",
            ),
            Self::UnknownEntry(entry) => write!(formatter, "no kernel entry named `{entry}` was found"),
            Self::AmbiguousEntries(entries) => write!(
                formatter,
                "multiple kernel entries were found ({}); select one with `--entry`",
                entries.join(", ")
            ),
            Self::UnsupportedBackend { kernel, backend, reason } => write!(
                formatter,
                "kernel `{kernel}` cannot use backend `{}`: {reason}",
                backend.name()
            ),
        }
    }
}

impl std::error::Error for KernelError {}

pub(crate) const fn tensor_element_name(element: TensorElementType) -> &'static str {
    element.name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_gpu_does_not_claim_triton_support() {
        assert!(!KernelTarget::Gpu.supports_triton());
        assert!(!KernelTarget::parse("nvidia").unwrap().supports_triton());
        assert!(KernelTarget::parse("cuda:sm_90").unwrap().supports_triton());
        assert!(KernelTarget::parse("cuda:sm_90a")
            .unwrap()
            .supports_triton());
        assert!(!KernelTarget::parse("cuda:sm_75").unwrap().supports_triton());
        assert!(KernelTarget::parse("rocm:gfx1100")
            .unwrap()
            .supports_triton());
    }
}
