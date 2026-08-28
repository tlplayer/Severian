#![deny(unsafe_code)]

//! Stable Severian-to-Triton compiler boundary.
//!
//! Triton's compiler core remains an external MLIR/C++ component. Severian
//! owns this versioned ABI, the full fusion graph, and the TTIR it submits.
//! Pass ordering is adapted from Triton (MIT); see `THIRD_PARTY_NOTICES.md`.

use severian_fusion::{
    AliasKind, ElementKind, FusionGraph, FusionRegion, GpuTarget, KernelSpecialization, Mutation,
    NodeKind, OperandRole, Rank, StorageLayout, Stride,
};
use std::collections::BTreeSet;
use std::fmt;

#[allow(unsafe_code)]
mod native;
mod ttir;

pub use native::NativeTritonCompiler;
pub use ttir::TtirModule;

pub const ABI_VERSION: u32 = 5;

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
            emit: KernelFormat::Ptx,
            debug: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledKernel {
    pub format: KernelFormat,
    pub entry_point: String,
    pub code: Vec<u8>,
    pub launch: LaunchMetadata,
}

impl CompiledKernel {
    pub fn shared_memory_bytes(&self) -> u64 {
        self.launch.shared_memory_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchMetadata {
    pub grid: [u64; 3],
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub shared_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    UnsupportedTarget,
    InvalidTtir(String),
    DonorCompiler(String),
    AbiMismatch { expected: u32, found: u32 },
    InvalidSpecialization(String),
    NativeUnavailable(String),
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
            Self::InvalidSpecialization(message) => {
                write!(formatter, "invalid kernel specialization: {message}")
            }
            Self::NativeUnavailable(message) => {
                write!(formatter, "native Triton bridge unavailable: {message}")
            }
        }
    }
}

impl std::error::Error for BridgeError {}

pub trait TritonCompiler: Send + Sync {
    fn compile(
        &self,
        graph: &FusionGraph,
        region: &FusionRegion,
        specialization: &KernelSpecialization,
        options: &CompileOptions,
    ) -> Result<CompiledKernel, BridgeError>;
}

pub fn lower_to_ttir(
    graph: &FusionGraph,
    region: &FusionRegion,
    specialization: &KernelSpecialization,
) -> Result<TtirModule, BridgeError> {
    specialization
        .validate_region(graph, region, specialization.target)
        .map_err(|error| BridgeError::InvalidSpecialization(error.to_string()))?;
    ttir::lower(graph, region, specialization).map_err(BridgeError::InvalidTtir)
}

impl From<GpuTarget> for CompileTarget {
    fn from(target: GpuTarget) -> Self {
        match target {
            GpuTarget::Amd => Self::AmdGpu,
            GpuTarget::Nvidia => Self::NvidiaGpu,
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiRankKind {
    Unranked = 0,
    Ranked = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiLayoutKind {
    Runtime = 0,
    Dense = 1,
    Strided = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiOperandRole {
    Data = 0,
    RuntimeShape = 1,
    RuntimeStrides = 2,
}

impl From<OperandRole> for AbiOperandRole {
    fn from(role: OperandRole) -> Self {
        match role {
            OperandRole::Data => Self::Data,
            OperandRole::RuntimeShape => Self::RuntimeShape,
            OperandRole::RuntimeStrides => Self::RuntimeStrides,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiAliasKind {
    View = 1,
    InPlace = 2,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiAlias {
    pub input_index: u16,
    pub _reserved: u16,
    pub kind: AbiAliasKind,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiBatchDimension {
    pub result: u32,
    pub lhs: i32,
    pub rhs: i32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiContractionDimension {
    pub lhs: u32,
    pub rhs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiMutationKind {
    None = 0,
    WritesInput = 1,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiNode {
    pub id: u32,
    pub kind: AbiNodeKind,
    pub operation: AbiBytes,
    pub attributes: AbiSlice<i64>,
    pub inputs: AbiSlice<u32>,
    pub operand_roles: AbiSlice<AbiOperandRole>,
    pub rank: AbiRankKind,
    /// Known dimensions are non-negative; `-1` is dynamic.
    pub dimensions: AbiSlice<i64>,
    pub layout: AbiLayoutKind,
    pub minor_to_major: AbiSlice<u32>,
    /// `i64::MIN` represents a dynamic stride or offset.
    pub strides: AbiSlice<i64>,
    pub layout_offset: i64,
    pub element_kind: AbiElementKind,
    pub element_bits: u16,
    pub _reserved: u16,
    pub aliases: AbiSlice<AbiAlias>,
    pub batch_dimensions: AbiSlice<AbiBatchDimension>,
    pub contraction_dimensions: AbiSlice<AbiContractionDimension>,
    pub mutation: AbiMutationKind,
    pub mutation_input: u16,
    pub _mutation_reserved: u16,
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
    pub members: AbiSlice<u32>,
    pub inputs: AbiSlice<u32>,
    pub outputs: AbiSlice<u32>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiRuntimeShape {
    pub node_id: u32,
    pub _reserved: u32,
    pub dimensions: AbiSlice<u64>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiRuntimeStrides {
    pub node_id: u32,
    pub _reserved: u32,
    pub strides: AbiSlice<i64>,
    pub offset: i64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiKernelSpecialization {
    pub target: CompileTarget,
    pub shapes: AbiSlice<AbiRuntimeShape>,
    pub strides: AbiSlice<AbiRuntimeStrides>,
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
    pub specialization: *const AbiKernelSpecialization,
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
    pub launch: AbiLaunchMetadata,
    pub owner: *mut std::ffi::c_void,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AbiLaunchMetadata {
    pub grid_x: u64,
    pub grid_y: u64,
    pub grid_z: u64,
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub _reserved: u32,
    pub shared_memory_bytes: u64,
}

pub type AbiCompileFn = unsafe extern "C" fn(
    request: *const AbiCompileRequest,
    output: *mut AbiCompiledKernel,
) -> AbiStatus;
pub type AbiDestroyKernelFn = unsafe extern "C" fn(kernel: *mut AbiCompiledKernel);

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
    specialization: &KernelSpecialization,
    options: &CompileOptions,
    callback: impl FnOnce(&AbiCompileRequest) -> R,
) -> Result<R, BridgeError> {
    let expected_target = match options.target {
        CompileTarget::AmdGpu => GpuTarget::Amd,
        CompileTarget::NvidiaGpu => GpuTarget::Nvidia,
    };
    specialization
        .validate_region(graph, region, expected_target)
        .map_err(|error| BridgeError::InvalidSpecialization(error.to_string()))?;
    let ttir = ttir::lower(graph, region, specialization).map_err(BridgeError::InvalidTtir)?;

    let descriptor_ids = region
        .inputs
        .iter()
        .chain(&region.nodes)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let input_storage = descriptor_ids
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
    let role_storage = descriptor_ids
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .operand_roles
                .iter()
                .copied()
                .map(AbiOperandRole::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dimension_storage = descriptor_ids
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
    let minor_to_major_storage = descriptor_ids
        .iter()
        .map(|id| match &graph.node(*id).layout {
            StorageLayout::Dense { minor_to_major } => minor_to_major.clone(),
            StorageLayout::Runtime | StorageLayout::Strided { .. } => Vec::new(),
        })
        .collect::<Vec<_>>();
    let stride_storage = descriptor_ids
        .iter()
        .map(|id| match &graph.node(*id).layout {
            StorageLayout::Strided { strides, .. } => strides
                .iter()
                .map(|stride| match stride {
                    Stride::Dynamic => i64::MIN,
                    Stride::Known(value) => *value,
                })
                .collect(),
            StorageLayout::Runtime | StorageLayout::Dense { .. } => Vec::new(),
        })
        .collect::<Vec<_>>();
    let alias_storage = descriptor_ids
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .aliases
                .iter()
                .map(|alias| AbiAlias {
                    input_index: alias.input_index,
                    _reserved: 0,
                    kind: match alias.kind {
                        AliasKind::View => AbiAliasKind::View,
                        AliasKind::InPlace => AbiAliasKind::InPlace,
                    },
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let batch_dimension_storage = descriptor_ids
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .matmul
                .as_ref()
                .map(|contract| {
                    contract
                        .batch_dimensions
                        .iter()
                        .map(|dimension| AbiBatchDimension {
                            result: dimension.result,
                            lhs: dimension.lhs.map_or(-1, |axis| axis as i32),
                            rhs: dimension.rhs.map_or(-1, |axis| axis as i32),
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect::<Vec<Vec<_>>>();
    let contraction_dimension_storage = descriptor_ids
        .iter()
        .map(|id| {
            graph
                .node(*id)
                .matmul
                .as_ref()
                .map(|contract| {
                    contract
                        .contraction_dimensions
                        .iter()
                        .map(|dimension| AbiContractionDimension {
                            lhs: dimension.lhs,
                            rhs: dimension.rhs,
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect::<Vec<Vec<_>>>();
    let nodes = descriptor_ids
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
                operand_roles: AbiSlice::from_slice(&role_storage[index]),
                rank: match node.shape.rank {
                    Rank::Unranked => AbiRankKind::Unranked,
                    Rank::Ranked(_) => AbiRankKind::Ranked,
                },
                dimensions: AbiSlice::from_slice(&dimension_storage[index]),
                layout: match node.layout {
                    StorageLayout::Runtime => AbiLayoutKind::Runtime,
                    StorageLayout::Dense { .. } => AbiLayoutKind::Dense,
                    StorageLayout::Strided { .. } => AbiLayoutKind::Strided,
                },
                minor_to_major: AbiSlice::from_slice(&minor_to_major_storage[index]),
                strides: AbiSlice::from_slice(&stride_storage[index]),
                layout_offset: match &node.layout {
                    StorageLayout::Strided { offset, .. } => match offset {
                        Stride::Dynamic => i64::MIN,
                        Stride::Known(value) => *value,
                    },
                    StorageLayout::Runtime | StorageLayout::Dense { .. } => 0,
                },
                element_kind: node.shape.element_kind.into(),
                element_bits: node.shape.element_bits,
                _reserved: 0,
                aliases: AbiSlice::from_slice(&alias_storage[index]),
                batch_dimensions: AbiSlice::from_slice(&batch_dimension_storage[index]),
                contraction_dimensions: AbiSlice::from_slice(&contraction_dimension_storage[index]),
                mutation: match node.mutation {
                    Mutation::None => AbiMutationKind::None,
                    Mutation::WritesInput(_) => AbiMutationKind::WritesInput,
                },
                mutation_input: match node.mutation {
                    Mutation::None => u16::MAX,
                    Mutation::WritesInput(input) => input,
                },
                _mutation_reserved: 0,
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
    let members = region.nodes.iter().map(|id| id.0).collect::<Vec<_>>();
    let abi_region = AbiFusionRegion {
        abi_version: ABI_VERSION,
        region_id: region.id.0,
        nodes: AbiSlice::from_slice(&nodes),
        members: AbiSlice::from_slice(&members),
        inputs: AbiSlice::from_slice(&inputs),
        outputs: AbiSlice::from_slice(&outputs),
    };
    let selected_shapes = specialization
        .shapes
        .iter()
        .filter(|shape| descriptor_ids.binary_search(&shape.node).is_ok())
        .collect::<Vec<_>>();
    let runtime_shape_storage = selected_shapes
        .iter()
        .map(|shape| shape.dimensions.clone())
        .collect::<Vec<_>>();
    let runtime_shapes = selected_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| AbiRuntimeShape {
            node_id: shape.node.0,
            _reserved: 0,
            dimensions: AbiSlice::from_slice(&runtime_shape_storage[index]),
        })
        .collect::<Vec<_>>();
    let selected_strides = specialization
        .strides
        .iter()
        .filter(|layout| descriptor_ids.binary_search(&layout.node).is_ok())
        .collect::<Vec<_>>();
    let runtime_stride_storage = selected_strides
        .iter()
        .map(|layout| layout.strides.clone())
        .collect::<Vec<_>>();
    let runtime_strides = selected_strides
        .iter()
        .enumerate()
        .map(|(index, layout)| AbiRuntimeStrides {
            node_id: layout.node.0,
            _reserved: 0,
            strides: AbiSlice::from_slice(&runtime_stride_storage[index]),
            offset: layout.offset,
        })
        .collect::<Vec<_>>();
    let abi_specialization = AbiKernelSpecialization {
        target: specialization.target.into(),
        shapes: AbiSlice::from_slice(&runtime_shapes),
        strides: AbiSlice::from_slice(&runtime_strides),
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
        specialization: &abi_specialization,
        ttir: AbiBytes::from_bytes(ttir.text.as_bytes()),
        options: &abi_options,
    };
    Ok(callback(&request))
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
    use severian_fusion::{
        plan, BatchDimension, ContractionDimension, DeviceModel, FusionNode, Matmul, NodeId, Shape,
    };
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    fn specialization() -> KernelSpecialization {
        KernelSpecialization {
            shapes: Vec::new(),
            strides: Vec::new(),
            target: GpuTarget::Amd,
        }
    }

    fn parse_with_pinned_triton(module: &TtirModule) {
        let Ok(parser) = std::env::var("SEVERIAN_TRITON_OPT") else {
            return;
        };
        let mut child = Command::new(parser)
            .arg("-verify-diagnostics")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(module.text.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "pinned Triton rejected {}:\n{}\n{}",
            module.entry_point,
            String::from_utf8_lossy(&output.stderr),
            module.text
        );
    }

    #[test]
    fn request_exposes_ttir_and_the_complete_selected_region() {
        let mut nodes = vec![
            FusionNode::structural(0, NodeKind::Parameter, [], Shape::ranked([8], 32)),
            FusionNode::structural(
                1,
                NodeKind::Elementwise,
                [NodeId(0)],
                Shape::ranked([8], 32),
            ),
        ];
        nodes[1].operation = "add".into();
        let graph = FusionGraph::new(nodes).unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        let options = CompileOptions::amd("gfx1100");
        let specialization = specialization();
        with_abi_request(
            &graph,
            &plan.regions[0],
            &specialization,
            &options,
            |request| {
                assert_eq!(request.abi_version, ABI_VERSION);
                assert!(request.ttir.len > 9);
                assert!(!request.region.is_null());
                assert!(!request.specialization.is_null());
                assert!(!request.options.is_null());
            },
        )
        .unwrap();
    }

    #[test]
    fn severian_owns_structural_ttir_construction() {
        let mut nodes = vec![
            FusionNode::structural(
                0,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(64)],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
            FusionNode::structural(
                1,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(64)],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
            FusionNode::structural(
                2,
                NodeKind::Elementwise,
                [NodeId(0), NodeId(1)],
                Shape::typed(
                    [severian_fusion::Dimension::Known(64)],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
        ];
        nodes[2].operation = "add".into();
        let graph = FusionGraph::new(nodes).unwrap();
        let broadcast_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module = lower_to_ttir(&graph, &broadcast_plan.regions[0], &specialization()).unwrap();
        assert_eq!(module.entry_point, "severian_region_0");
        assert!(module.text.contains("tt.get_program_id"));
        assert!(module.text.contains("arith.cmpi slt"));
        assert!(module.text.contains("tt.load"));
        assert!(module.text.contains("arith.addf"));
        assert!(module.text.contains("tt.store"));
        parse_with_pinned_triton(&module);
    }

    #[test]
    fn every_pure_structural_class_emits_parseable_ttir() {
        let f32_shape = |dimensions: &[u64]| {
            Shape::typed(
                dimensions
                    .iter()
                    .copied()
                    .map(severian_fusion::Dimension::Known),
                ElementKind::IeeeFloat,
                32,
            )
        };

        let cases = [
            (NodeKind::Reshape, "reshape", vec![8, 8], vec![8, 8]),
            (NodeKind::StorageView, "materialize", vec![8, 8], vec![8, 8]),
            (NodeKind::Permute, "reverse", vec![4, 8], vec![8, 4]),
        ];
        for (kind, operation, input_shape, output_shape) in cases {
            let mut operation_node =
                FusionNode::structural(1, kind, [NodeId(0)], f32_shape(&output_shape));
            operation_node.operation = operation.into();
            let graph = FusionGraph::new(vec![
                FusionNode::structural(0, NodeKind::Parameter, [], f32_shape(&input_shape)),
                operation_node,
            ])
            .unwrap();
            let plan = plan(&graph, DeviceModel::conservative_gpu());
            let module = lower_to_ttir(&graph, &plan.regions[0], &specialization()).unwrap();
            assert!(module.text.contains(&format!("severian.{kind:?}")));
            assert!(module.text.contains("arith.cmpi slt"));
            parse_with_pinned_triton(&module);
        }

        let mut slice = FusionNode::structural(1, NodeKind::Slice, [NodeId(0)], f32_shape(&[4, 4]));
        slice.operation = "slice".into();
        slice.layout = StorageLayout::Strided {
            strides: vec![Stride::Known(16), Stride::Known(2)],
            offset: Stride::Known(0),
        };
        let graph = FusionGraph::new(vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape(&[8, 8])),
            slice,
        ])
        .unwrap();
        let slice_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module = lower_to_ttir(&graph, &slice_plan.regions[0], &specialization()).unwrap();
        assert!(module.text.contains("Slice.input_0_strided_index"));
        parse_with_pinned_triton(&module);

        let mut gather = FusionNode::structural(
            2,
            NodeKind::Gather,
            [NodeId(0), NodeId(1)],
            f32_shape(&[64]),
        );
        gather.operation = "gather".into();
        let graph = FusionGraph::new(vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape(&[64])),
            FusionNode::structural(
                1,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(64)],
                    ElementKind::SignedInteger,
                    32,
                ),
            ),
            gather,
        ])
        .unwrap();
        let gather_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module = lower_to_ttir(&graph, &gather_plan.regions[0], &specialization()).unwrap();
        assert!(module.text.contains("Gather.indexed_masked_load"));
        parse_with_pinned_triton(&module);

        let mut broadcast =
            FusionNode::structural(1, NodeKind::Broadcast, [NodeId(0)], f32_shape(&[8, 8]));
        broadcast.operation = "like".into();
        let graph = FusionGraph::new(vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape(&[8, 1])),
            broadcast,
        ])
        .unwrap();
        let broadcast_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module = lower_to_ttir(&graph, &broadcast_plan.regions[0], &specialization()).unwrap();
        assert!(module.text.contains("row_major_index"));
        parse_with_pinned_triton(&module);

        let mut convert =
            FusionNode::structural(1, NodeKind::Convert, [NodeId(0)], f32_shape(&[64]));
        convert.operation = "convert".into();
        let graph = FusionGraph::new(vec![
            FusionNode::structural(
                0,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(64)],
                    ElementKind::IeeeFloat,
                    16,
                ),
            ),
            convert,
        ])
        .unwrap();
        let convert_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module = lower_to_ttir(&graph, &convert_plan.regions[0], &specialization()).unwrap();
        assert!(module.text.contains("arith.extf"));
        parse_with_pinned_triton(&module);

        let mut concatenate = FusionNode::structural(
            3,
            NodeKind::Concatenate,
            [NodeId(0), NodeId(1), NodeId(2)],
            f32_shape(&[2, 8]),
        );
        concatenate.operation = "concatenate".into();
        concatenate.operand_roles = vec![
            OperandRole::Data,
            OperandRole::Data,
            OperandRole::RuntimeShape,
        ];
        let graph = FusionGraph::new(vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape(&[2, 3])),
            FusionNode::structural(1, NodeKind::Parameter, [], f32_shape(&[2, 5])),
            FusionNode::structural(2, NodeKind::Parameter, [], Shape::ranked([1], 64)),
            concatenate,
        ])
        .unwrap();
        let concatenate_plan = plan(&graph, DeviceModel::conservative_gpu());
        let module =
            lower_to_ttir(&graph, &concatenate_plan.regions[0], &specialization()).unwrap();
        assert!(module
            .text
            .contains("severian.Concatenate.indexed_masked_select"));
        parse_with_pinned_triton(&module);
    }

    #[test]
    fn reduction_matmul_and_effect_boundaries_have_owned_emitters() {
        let f32_shape = || {
            Shape::typed(
                [severian_fusion::Dimension::Known(256)],
                ElementKind::IeeeFloat,
                32,
            )
        };
        let mut reduction_nodes = vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape()),
            FusionNode::structural(
                1,
                NodeKind::Reduction,
                [NodeId(0)],
                Shape::typed(
                    [severian_fusion::Dimension::Known(1)],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
        ];
        reduction_nodes[1].operation = "sum".into();
        let reduction_graph = FusionGraph::new(reduction_nodes).unwrap();
        let reduction_plan = plan(&reduction_graph, DeviceModel::conservative_gpu());
        let reduction = lower_to_ttir(
            &reduction_graph,
            &reduction_plan.regions[0],
            &specialization(),
        )
        .unwrap();
        assert!(reduction.text.contains("\"tt.reduce\""));
        parse_with_pinned_triton(&reduction);

        let mut fused_reduction_nodes = vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape()),
            FusionNode::structural(
                1,
                NodeKind::Elementwise,
                [NodeId(0), NodeId(0)],
                f32_shape(),
            ),
            FusionNode::structural(
                2,
                NodeKind::Reduction,
                [NodeId(1)],
                Shape::typed(
                    [severian_fusion::Dimension::Known(1)],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
            FusionNode::structural(
                3,
                NodeKind::Elementwise,
                [NodeId(0), NodeId(2)],
                f32_shape(),
            ),
        ];
        fused_reduction_nodes[1].operation = "multiply".into();
        fused_reduction_nodes[2].operation = "mean_last".into();
        fused_reduction_nodes[3].operation = "multiply".into();
        let fused_reduction_graph = FusionGraph::new(fused_reduction_nodes).unwrap();
        let fused_reduction_plan = plan(&fused_reduction_graph, DeviceModel::conservative_gpu());
        assert_eq!(fused_reduction_plan.regions.len(), 1);
        let fused_reduction = lower_to_ttir(
            &fused_reduction_graph,
            &fused_reduction_plan.regions[0],
            &specialization(),
        )
        .unwrap();
        assert_eq!(fused_reduction.text.matches("arith.mulf").count(), 2);
        assert!(fused_reduction.text.contains("arith.divf"));
        parse_with_pinned_triton(&fused_reduction);

        let rank = Rank::Ranked(vec![severian_fusion::Dimension::Known(16); 2]);
        let mut matmul_nodes = vec![
            FusionNode::structural(
                0,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(16); 2],
                    ElementKind::IeeeFloat,
                    16,
                ),
            ),
            FusionNode::structural(
                1,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(16); 2],
                    ElementKind::IeeeFloat,
                    16,
                ),
            ),
            FusionNode::structural(
                2,
                NodeKind::Contraction,
                [NodeId(0), NodeId(1)],
                Shape::typed(
                    [severian_fusion::Dimension::Known(16); 2],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
        ];
        matmul_nodes[2].operation = "matmul".into();
        matmul_nodes[2].matmul = Some(Matmul {
            lhs_shape: rank.clone(),
            rhs_shape: rank.clone(),
            result_shape: rank,
            batch_dimensions: Vec::<BatchDimension>::new(),
            contraction_dimensions: vec![ContractionDimension { lhs: 1, rhs: 0 }],
        });
        let matmul_graph = FusionGraph::new(matmul_nodes).unwrap();
        let matmul_plan = plan(&matmul_graph, DeviceModel::conservative_gpu());
        let matmul =
            lower_to_ttir(&matmul_graph, &matmul_plan.regions[0], &specialization()).unwrap();
        assert!(matmul.text.contains("tt.dot"));
        assert!(!matmul.entry_point.contains("rank"));
        assert!(!matmul.entry_point.contains("f16"));
        parse_with_pinned_triton(&matmul);

        let rank4 = Rank::Ranked(vec![
            severian_fusion::Dimension::Known(2),
            severian_fusion::Dimension::Known(3),
            severian_fusion::Dimension::Known(16),
            severian_fusion::Dimension::Known(16),
        ]);
        let mut batched_nodes = vec![
            FusionNode::structural(
                0,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [
                        severian_fusion::Dimension::Known(2),
                        severian_fusion::Dimension::Known(3),
                        severian_fusion::Dimension::Known(16),
                        severian_fusion::Dimension::Known(16),
                    ],
                    ElementKind::IeeeFloat,
                    16,
                ),
            ),
            FusionNode::structural(
                1,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [
                        severian_fusion::Dimension::Known(2),
                        severian_fusion::Dimension::Known(3),
                        severian_fusion::Dimension::Known(16),
                        severian_fusion::Dimension::Known(16),
                    ],
                    ElementKind::IeeeFloat,
                    16,
                ),
            ),
            FusionNode::structural(
                2,
                NodeKind::Contraction,
                [NodeId(0), NodeId(1)],
                Shape::typed(
                    [
                        severian_fusion::Dimension::Known(2),
                        severian_fusion::Dimension::Known(3),
                        severian_fusion::Dimension::Known(16),
                        severian_fusion::Dimension::Known(16),
                    ],
                    ElementKind::IeeeFloat,
                    32,
                ),
            ),
        ];
        batched_nodes[2].operation = "matmul".into();
        batched_nodes[2].matmul = Some(Matmul {
            lhs_shape: rank4.clone(),
            rhs_shape: rank4.clone(),
            result_shape: rank4,
            batch_dimensions: vec![
                BatchDimension {
                    result: 0,
                    lhs: Some(0),
                    rhs: Some(0),
                },
                BatchDimension {
                    result: 1,
                    lhs: Some(1),
                    rhs: Some(1),
                },
            ],
            contraction_dimensions: vec![ContractionDimension { lhs: 3, rhs: 2 }],
        });
        let batched_graph = FusionGraph::new(batched_nodes).unwrap();
        let batched_plan = plan(&batched_graph, DeviceModel::conservative_gpu());
        let batched =
            lower_to_ttir(&batched_graph, &batched_plan.regions[0], &specialization()).unwrap();
        assert_eq!(batched.entry_point, matmul.entry_point);
        assert!(batched.text.contains("BatchDimension"));
        assert!(!batched.text.contains("matmul_rank4"));
        parse_with_pinned_triton(&batched);

        let mut scatter_nodes = vec![
            FusionNode::structural(0, NodeKind::Parameter, [], f32_shape()),
            FusionNode::structural(
                1,
                NodeKind::Parameter,
                [],
                Shape::typed(
                    [severian_fusion::Dimension::Known(256)],
                    ElementKind::SignedInteger,
                    32,
                ),
            ),
            FusionNode::structural(2, NodeKind::Parameter, [], f32_shape()),
            FusionNode::structural(
                3,
                NodeKind::Scatter,
                [NodeId(0), NodeId(1), NodeId(2)],
                f32_shape(),
            ),
        ];
        scatter_nodes[3].operation = "scatter".into();
        scatter_nodes[3].mutation = Mutation::WritesInput(0);
        let scatter_graph = FusionGraph::new(scatter_nodes).unwrap();
        let scatter_plan = plan(&scatter_graph, DeviceModel::conservative_gpu());
        let scatter =
            lower_to_ttir(&scatter_graph, &scatter_plan.regions[0], &specialization()).unwrap();
        assert!(scatter
            .text
            .contains("severian.Scatter.indexed_masked_store"));
        parse_with_pinned_triton(&scatter);
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
