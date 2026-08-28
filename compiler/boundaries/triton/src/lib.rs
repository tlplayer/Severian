#![forbid(unsafe_code)]

//! Stable Severian-to-Triton compiler boundary.
//!
//! Triton's compiler core remains an external MLIR/C++ component. Severian
//! owns this versioned ABI, the full fusion graph, and the TTIR it submits.
//! Pass ordering is adapted from Triton (MIT); see `THIRD_PARTY_NOTICES.md`.

use severian_fusion::{ElementKind, FusionGraph, FusionRegion, NodeKind};
use std::fmt;

pub const ABI_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CompileTarget {
    AmdGpu = 1,
    NvidiaGpu = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum KernelFormat {
    LlvmIr = 1,
    AmdGcN = 2,
    Hsaco = 3,
    Ptx = 4,
    Cubin = 5,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    pub target: CompileTarget,
    pub architecture: String,
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub num_stages: u32,
    pub emit: KernelFormat,
    pub debug: bool,
}

impl CompileOptions {
    pub fn amd(architecture: impl Into<String>) -> Self {
        Self {
            target: CompileTarget::AmdGpu,
            architecture: architecture.into(),
            num_warps: 4,
            warp_size: 64,
            num_ctas: 1,
            num_stages: 2,
            emit: KernelFormat::Hsaco,
            debug: false,
        }
    }

    pub fn nvidia(architecture: impl Into<String>) -> Self {
        Self {
            target: CompileTarget::NvidiaGpu,
            architecture: architecture.into(),
            num_warps: 4,
            warp_size: 32,
            num_ctas: 1,
            num_stages: 3,
            emit: KernelFormat::Cubin,
            debug: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledKernel {
    pub format: KernelFormat,
    pub entry_point: String,
    pub code: Vec<u8>,
    pub shared_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    UnsupportedTarget,
    InvalidTtir(String),
    DonorCompiler(String),
    AbiMismatch { expected: u32, found: u32 },
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTarget => {
                formatter.write_str("Triton supports AMD and NVIDIA GPU targets only")
            }
            Self::InvalidTtir(message) => write!(formatter, "invalid TTIR: {message}"),
            Self::DonorCompiler(message) => write!(formatter, "Triton compiler failed: {message}"),
            Self::AbiMismatch { expected, found } => write!(
                formatter,
                "Triton bridge ABI mismatch: expected {expected}, found {found}"
            ),
        }
    }
}

impl std::error::Error for BridgeError {}

pub trait TritonCompiler: Send + Sync {
    fn compile(
        &self,
        graph: &FusionGraph,
        region: &FusionRegion,
        ttir: &str,
        options: &CompileOptions,
    ) -> Result<CompiledKernel, BridgeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AbiStatus {
    Ok = 0,
    InvalidArgument = 1,
    ParseFailure = 2,
    PassFailure = 3,
    CodegenFailure = 4,
    UnsupportedTarget = 5,
    InternalFailure = 255,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiSlice<T> {
    pub data: *const T,
    pub len: usize,
}

impl<T> AbiSlice<T> {
    fn from_slice(values: &[T]) -> Self {
        Self {
            data: values.as_ptr(),
            len: values.len(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiBytes {
    pub data: *const u8,
    pub len: usize,
}

impl AbiBytes {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiNodeKind {
    Parameter = 0,
    Constant = 1,
    Elementwise = 2,
    Reduction = 3,
    Contraction = 4,
    Reshape = 5,
    Permute = 6,
    Slice = 7,
    Broadcast = 8,
    Gather = 9,
    Scatter = 10,
    Concatenate = 11,
    Convert = 12,
    StorageView = 13,
}

impl From<NodeKind> for AbiNodeKind {
    fn from(kind: NodeKind) -> Self {
        match kind {
            NodeKind::Parameter => Self::Parameter,
            NodeKind::Constant => Self::Constant,
            NodeKind::Elementwise => Self::Elementwise,
            NodeKind::Reduction => Self::Reduction,
            NodeKind::Contraction => Self::Contraction,
            NodeKind::Reshape => Self::Reshape,
            NodeKind::Permute => Self::Permute,
            NodeKind::Slice => Self::Slice,
            NodeKind::Broadcast => Self::Broadcast,
            NodeKind::Gather => Self::Gather,
            NodeKind::Scatter => Self::Scatter,
            NodeKind::Concatenate => Self::Concatenate,
            NodeKind::Convert => Self::Convert,
            NodeKind::StorageView => Self::StorageView,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiElementKind {
    SignedInteger = 1,
    UnsignedInteger = 2,
    IeeeFloat = 3,
    BrainFloat = 4,
    Float8E4M3Fn = 5,
    Float8E5M2 = 6,
    Boolean = 7,
    Opaque = 255,
}

impl From<ElementKind> for AbiElementKind {
    fn from(kind: ElementKind) -> Self {
        match kind {
            ElementKind::SignedInteger => Self::SignedInteger,
            ElementKind::UnsignedInteger => Self::UnsignedInteger,
            ElementKind::IeeeFloat => Self::IeeeFloat,
            ElementKind::BrainFloat => Self::BrainFloat,
            ElementKind::Float8E4M3Fn => Self::Float8E4M3Fn,
            ElementKind::Float8E5M2 => Self::Float8E5M2,
            ElementKind::Boolean => Self::Boolean,
            ElementKind::Opaque => Self::Opaque,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiNode {
    pub id: u32,
    pub kind: AbiNodeKind,
    pub operation: AbiBytes,
    pub attributes: AbiSlice<i64>,
    pub inputs: AbiSlice<u32>,
    /// Known dimensions are non-negative; `-1` is dynamic.
    pub dimensions: AbiSlice<i64>,
    pub element_kind: AbiElementKind,
    pub element_bits: u16,
    pub _reserved: u16,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub flops: u64,
    pub shared_memory_bytes: u64,
    pub unnested_reductions: u16,
    pub has_side_effects: u8,
    pub _padding: [u8; 5],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiFusionRegion {
    pub abi_version: u32,
    pub region_id: u32,
    pub nodes: AbiSlice<AbiNode>,
    pub inputs: AbiSlice<u32>,
    pub outputs: AbiSlice<u32>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiCompileOptions {
    pub target: CompileTarget,
    pub architecture: AbiBytes,
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub num_stages: u32,
    pub emit: KernelFormat,
    pub debug: u8,
    pub _padding: [u8; 7],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiCompileRequest {
    pub abi_version: u32,
    pub region: *const AbiFusionRegion,
    pub ttir: AbiBytes,
    pub options: *const AbiCompileOptions,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiCompiledKernel {
    pub abi_version: u32,
    pub format: KernelFormat,
    pub entry_point: AbiBytes,
    pub code: AbiBytes,
    pub diagnostics: AbiBytes,
    pub shared_memory_bytes: u64,
    pub owner: *mut std::ffi::c_void,
}

pub type AbiCompileFn =
    extern "C" fn(request: *const AbiCompileRequest, output: *mut AbiCompiledKernel) -> AbiStatus;
pub type AbiDestroyKernelFn = extern "C" fn(kernel: *mut AbiCompiledKernel);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct AbiBridgeV1 {
    pub abi_version: u32,
    pub compile: Option<AbiCompileFn>,
    pub destroy_kernel: Option<AbiDestroyKernelFn>,
}

/// Builds pointer-bearing ABI views and keeps all backing storage alive only
/// for the duration of `callback`. The views cannot escape this function.
pub fn with_abi_request<R>(
    graph: &FusionGraph,
    region: &FusionRegion,
    ttir: &str,
    options: &CompileOptions,
    callback: impl FnOnce(&AbiCompileRequest) -> R,
) -> R {
    let input_storage = region
        .nodes
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .inputs
                .iter()
                .map(|input| input.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dimension_storage = region
        .nodes
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .shape
                .dimensions()
                .unwrap_or_default()
                .iter()
                .map(|dimension| match dimension {
                    severian_fusion::Dimension::Dynamic => -1,
                    severian_fusion::Dimension::Known(value) => {
                        i64::try_from(*value).unwrap_or(i64::MAX)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let nodes = region
        .nodes
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let node = graph.node(*id);
            AbiNode {
                id: id.0,
                kind: node.kind.into(),
                operation: AbiBytes::from_bytes(node.operation.as_bytes()),
                attributes: AbiSlice::from_slice(&node.attributes),
                inputs: AbiSlice::from_slice(&input_storage[index]),
                dimensions: AbiSlice::from_slice(&dimension_storage[index]),
                element_kind: node.shape.element_kind.into(),
                element_bits: node.shape.element_bytes.saturating_mul(8),
                _reserved: 0,
                bytes_read: node.bytes_read,
                bytes_written: node.bytes_written,
                flops: node.flops,
                shared_memory_bytes: node.shared_memory_bytes,
                unnested_reductions: node.unnested_reductions,
                has_side_effects: u8::from(node.has_side_effects),
                _padding: [0; 5],
            }
        })
        .collect::<Vec<_>>();
    let inputs = region.inputs.iter().map(|id| id.0).collect::<Vec<_>>();
    let outputs = region.outputs.iter().map(|id| id.0).collect::<Vec<_>>();
    let abi_region = AbiFusionRegion {
        abi_version: ABI_VERSION,
        region_id: region.id.0,
        nodes: AbiSlice::from_slice(&nodes),
        inputs: AbiSlice::from_slice(&inputs),
        outputs: AbiSlice::from_slice(&outputs),
    };
    let abi_options = AbiCompileOptions {
        target: options.target,
        architecture: AbiBytes::from_bytes(options.architecture.as_bytes()),
        num_warps: options.num_warps,
        warp_size: options.warp_size,
        num_ctas: options.num_ctas,
        num_stages: options.num_stages,
        emit: options.emit,
        debug: u8::from(options.debug),
        _padding: [0; 7],
    };
    let request = AbiCompileRequest {
        abi_version: ABI_VERSION,
        region: &abi_region,
        ttir: AbiBytes::from_bytes(ttir.as_bytes()),
        options: &abi_options,
    };
    callback(&request)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassStage {
    Ttir,
    TritonGpu,
    LlvmDialect,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassSpec {
    pub stage: PassStage,
    pub donor_name: &'static str,
}

/// Severian-owned transcription of the stable pass ordering that is currently
/// orchestrated by Triton's AMD/NVIDIA `backend/compiler.py` files. The native
/// bridge maps each name to the donor pass constructor; Python is not part of
/// the production boundary.
pub fn pass_pipeline(target: CompileTarget) -> Vec<PassSpec> {
    let mut passes = vec![
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "inliner",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "canonicalizer",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "triton-combine",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "reorder-broadcast",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "cse",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "triton-licm",
        },
        PassSpec {
            stage: PassStage::Ttir,
            donor_name: "loop-unroll",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "convert-to-ttgpuir",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "coalesce",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "remove-layout-conversions",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "optimize-thread-locality",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "accelerate-matmul",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "optimize-epilogue",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "schedule-loops",
        },
        PassSpec {
            stage: PassStage::TritonGpu,
            donor_name: "pipeline",
        },
        PassSpec {
            stage: PassStage::LlvmDialect,
            donor_name: "allocate-shared-memory",
        },
    ];
    passes.extend(match target {
        CompileTarget::AmdGpu => [
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "tritongpu-to-amd-llvm",
            },
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "amd-warp-specialize-to-llvm",
            },
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "cf-to-llvm",
            },
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "arith-to-llvm",
            },
            PassSpec {
                stage: PassStage::Binary,
                donor_name: "llvm-to-amdgcn",
            },
            PassSpec {
                stage: PassStage::Binary,
                donor_name: "amdgcn-to-hsaco",
            },
        ],
        CompileTarget::NvidiaGpu => [
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "tritongpu-to-nvidia-llvm",
            },
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "nvgpu-to-llvm",
            },
            PassSpec {
                stage: PassStage::LlvmDialect,
                donor_name: "nvvm-to-llvm",
            },
            PassSpec {
                stage: PassStage::Binary,
                donor_name: "llvm-to-ptx",
            },
            PassSpec {
                stage: PassStage::Binary,
                donor_name: "ptx-to-cubin",
            },
            PassSpec {
                stage: PassStage::Binary,
                donor_name: "cubin-metadata",
            },
        ],
    });
    passes
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_fusion::{plan, DeviceModel, FusionNode, NodeId, Shape};

    #[test]
    fn request_exposes_ttir_and_the_complete_selected_region() {
        let graph = FusionGraph::new(vec![
            FusionNode::structural(0, NodeKind::Parameter, [], Shape::ranked([8], 4)),
            FusionNode::structural(1, NodeKind::Elementwise, [NodeId(0)], Shape::ranked([8], 4)),
        ])
        .unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        let options = CompileOptions::amd("gfx1100");
        with_abi_request(&graph, &plan.regions[0], "module {}", &options, |request| {
            assert_eq!(request.abi_version, ABI_VERSION);
            assert_eq!(request.ttir.len, 9);
            assert!(!request.region.is_null());
            assert!(!request.options.is_null());
        });
    }

    #[test]
    fn donor_pipeline_is_target_specific_after_shared_ttir_stages() {
        let amd = pass_pipeline(CompileTarget::AmdGpu);
        let nvidia = pass_pipeline(CompileTarget::NvidiaGpu);
        assert_eq!(amd[0], nvidia[0]);
        assert!(amd.iter().any(|pass| pass.donor_name == "amdgcn-to-hsaco"));
        assert!(nvidia.iter().any(|pass| pass.donor_name == "ptx-to-cubin"));
        assert!(!amd.iter().any(|pass| pass.donor_name == "ptx-to-cubin"));
    }

    #[test]
    fn dtype_kind_is_abi_data_not_a_function_name() {
        assert_eq!(
            AbiElementKind::from(ElementKind::IeeeFloat),
            AbiElementKind::IeeeFloat
        );
        assert_ne!(
            AbiElementKind::from(ElementKind::SignedInteger),
            AbiElementKind::from(ElementKind::UnsignedInteger)
        );
    }
}
