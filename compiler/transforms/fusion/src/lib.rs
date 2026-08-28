#![forbid(unsafe_code)]

//! Device-neutral tensor graph fusion.
//!
//! The priority-queue policy and budget vocabulary are adapted from XLA:GPU's
//! `PriorityFusion` and `gpu_fusible` work (Apache-2.0). This is a Severian
//! graph implementation: it has no HLO types and does not import XLA's pass
//! manager or frontend.
//!
//! Copyright 2017-2018 The OpenXLA Authors. Donor portions are licensed under
//! Apache-2.0; see `THIRD_PARTY_NOTICES.md` at the repository root.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RegionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Dynamic,
    Known(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rank {
    Unranked,
    Ranked(Vec<Dimension>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stride {
    Dynamic,
    Known(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageLayout {
    /// Physical strides and offset are supplied when a kernel is specialized.
    Runtime,
    Dense {
        minor_to_major: Vec<u32>,
    },
    Strided {
        strides: Vec<Stride>,
        offset: Stride,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandRole {
    Data,
    RuntimeShape,
    RuntimeStrides,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    View,
    InPlace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputAlias {
    pub input_index: u16,
    pub kind: AliasKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    WritesInput(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTarget {
    Amd,
    Nvidia,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeShape {
    pub node: NodeId,
    pub dimensions: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStrides {
    pub node: NodeId,
    pub strides: Vec<i64>,
    pub offset: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSpecialization {
    pub shapes: Vec<RuntimeShape>,
    pub strides: Vec<RuntimeStrides>,
    pub target: GpuTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchDimension {
    pub result: u32,
    pub lhs: Option<u32>,
    pub rhs: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContractionDimension {
    pub lhs: u32,
    pub rhs: u32,
}

/// Rank-generic contraction metadata. Rank and dtype remain ordinary data;
/// this record never changes the `Matmul` operation identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matmul {
    pub lhs_shape: Rank,
    pub rhs_shape: Rank,
    pub result_shape: Rank,
    pub batch_dimensions: Vec<BatchDimension>,
    pub contraction_dimensions: Vec<ContractionDimension>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    SignedInteger,
    UnsignedInteger,
    IeeeFloat,
    BrainFloat,
    Float8E4M3Fn,
    Float8E5M2,
    Boolean,
    Opaque,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub rank: Rank,
    pub element_kind: ElementKind,
    pub element_bits: u16,
}

impl Shape {
    pub fn ranked(dimensions: impl IntoIterator<Item = u64>, element_bits: u16) -> Self {
        Self {
            rank: Rank::Ranked(dimensions.into_iter().map(Dimension::Known).collect()),
            element_kind: ElementKind::Opaque,
            element_bits,
        }
    }

    pub fn typed(
        dimensions: impl IntoIterator<Item = Dimension>,
        element_kind: ElementKind,
        element_bits: u16,
    ) -> Self {
        Self {
            rank: Rank::Ranked(dimensions.into_iter().collect()),
            element_kind,
            element_bits,
        }
    }

    pub fn unranked(element_kind: ElementKind, element_bits: u16) -> Self {
        Self {
            rank: Rank::Unranked,
            element_kind,
            element_bits,
        }
    }

    pub fn dimensions(&self) -> Option<&[Dimension]> {
        match &self.rank {
            Rank::Unranked => None,
            Rank::Ranked(dimensions) => Some(dimensions),
        }
    }

    pub fn byte_size(&self) -> Option<u64> {
        self.dimensions()?
            .iter()
            .try_fold(u64::from(self.element_bytes()), |bytes, dimension| {
                let Dimension::Known(dimension) = dimension else {
                    return None;
                };
                bytes.checked_mul(*dimension)
            })
    }

    pub const fn element_bytes(&self) -> u16 {
        self.element_bits.div_ceil(8)
    }

    pub fn default_layout(&self) -> StorageLayout {
        match &self.rank {
            Rank::Unranked => StorageLayout::Runtime,
            Rank::Ranked(dimensions) => StorageLayout::Dense {
                minor_to_major: (0..dimensions.len() as u32).rev().collect(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Parameter,
    Constant,
    Elementwise,
    Reduction,
    Contraction,
    Reshape,
    Permute,
    Slice,
    Broadcast,
    Gather,
    Scatter,
    Concatenate,
    Convert,
    StorageView,
}

impl NodeKind {
    pub const fn is_elementwise(self) -> bool {
        matches!(self, Self::Elementwise | Self::Convert)
    }

    pub const fn is_reduction(self) -> bool {
        matches!(self, Self::Reduction)
    }

    pub const fn is_contraction(self) -> bool {
        matches!(self, Self::Contraction)
    }

    const fn is_source(self) -> bool {
        matches!(self, Self::Parameter | Self::Constant)
    }

    const fn hero(self) -> Option<FusionHero> {
        match self {
            Self::Reduction => Some(FusionHero::Reduction),
            Self::Contraction => Some(FusionHero::Contraction),
            Self::Permute => Some(FusionHero::Transpose),
            Self::Gather => Some(FusionHero::Gather),
            Self::Scatter => Some(FusionHero::Scatter),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionNode {
    pub id: NodeId,
    pub kind: NodeKind,
    /// Stable Severian opcode spelling (`multiply`, `reduce_sum`, `matmul`, ...).
    /// The class in `kind` drives eligibility; this spelling drives TTIR emission.
    pub operation: String,
    /// Operation-specific integral data such as axes or a permutation.
    pub attributes: Vec<i64>,
    pub inputs: Vec<NodeId>,
    pub operand_roles: Vec<OperandRole>,
    pub shape: Shape,
    pub layout: StorageLayout,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub flops: u64,
    pub shared_memory_bytes: u64,
    pub unnested_reductions: u16,
    pub has_side_effects: bool,
    pub aliases: Vec<InputAlias>,
    pub mutation: Mutation,
    pub matmul: Option<Matmul>,
}

impl FusionNode {
    pub fn structural(
        id: u32,
        kind: NodeKind,
        inputs: impl IntoIterator<Item = NodeId>,
        shape: Shape,
    ) -> Self {
        let bytes = shape.byte_size().unwrap_or(0);
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let layout = shape.default_layout();
        Self {
            id: NodeId(id),
            kind,
            operation: format!("{kind:?}").to_ascii_lowercase(),
            attributes: Vec::new(),
            operand_roles: vec![OperandRole::Data; inputs.len()],
            inputs,
            shape,
            layout,
            bytes_read: bytes,
            bytes_written: bytes,
            flops: 0,
            shared_memory_bytes: 0,
            unnested_reductions: u16::from(kind.is_reduction()),
            has_side_effects: false,
            aliases: Vec::new(),
            mutation: Mutation::None,
            matmul: None,
        }
    }

    pub fn runtime_shape_inputs(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inputs
            .iter()
            .copied()
            .zip(&self.operand_roles)
            .filter_map(|(input, role)| {
                matches!(
                    role,
                    OperandRole::RuntimeShape | OperandRole::RuntimeStrides
                )
                .then_some(input)
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    NonContiguousId { expected: NodeId, found: NodeId },
    ForwardReference { node: NodeId, input: NodeId },
    OperandRoleCount { node: NodeId },
    InvalidLayout { node: NodeId },
    InvalidAliasInput { node: NodeId, input_index: u16 },
    InvalidMutationInput { node: NodeId, input_index: u16 },
    InvalidMatmulContract { node: NodeId },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonContiguousId { expected, found } => write!(
                formatter,
                "fusion node ids must be contiguous: expected {}, found {}",
                expected.0, found.0
            ),
            Self::ForwardReference { node, input } => write!(
                formatter,
                "fusion node {} references non-preceding node {}",
                node.0, input.0
            ),
            Self::OperandRoleCount { node } => write!(
                formatter,
                "fusion node {} has a different number of inputs and operand roles",
                node.0
            ),
            Self::InvalidLayout { node } => {
                write!(
                    formatter,
                    "fusion node {} has a layout incompatible with its rank",
                    node.0
                )
            }
            Self::InvalidAliasInput { node, input_index } => write!(
                formatter,
                "fusion node {} aliases missing input {}",
                node.0, input_index
            ),
            Self::InvalidMutationInput { node, input_index } => write!(
                formatter,
                "fusion node {} mutates missing input {}",
                node.0, input_index
            ),
            Self::InvalidMatmulContract { node } => write!(
                formatter,
                "fusion node {} has invalid rank-generic matmul metadata",
                node.0
            ),
        }
    }
}

impl std::error::Error for GraphError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionGraph {
    nodes: Vec<FusionNode>,
    users: Vec<Vec<NodeId>>,
}

impl FusionGraph {
    pub fn new(nodes: Vec<FusionNode>) -> Result<Self, GraphError> {
        let mut users = vec![Vec::new(); nodes.len()];
        for (index, node) in nodes.iter().enumerate() {
            let expected = NodeId(index as u32);
            if node.id != expected {
                return Err(GraphError::NonContiguousId {
                    expected,
                    found: node.id,
                });
            }
            if node.inputs.len() != node.operand_roles.len() {
                return Err(GraphError::OperandRoleCount { node: node.id });
            }
            let valid_layout = match (&node.shape.rank, &node.layout) {
                (Rank::Unranked, StorageLayout::Runtime) => true,
                (Rank::Unranked, StorageLayout::Dense { .. } | StorageLayout::Strided { .. }) => {
                    false
                }
                (Rank::Ranked(_), StorageLayout::Runtime) => true,
                (Rank::Ranked(dimensions), StorageLayout::Dense { minor_to_major }) => {
                    minor_to_major.len() == dimensions.len()
                        && minor_to_major
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>()
                            .len()
                            == dimensions.len()
                        && minor_to_major
                            .iter()
                            .all(|axis| (*axis as usize) < dimensions.len())
                }
                (Rank::Ranked(dimensions), StorageLayout::Strided { strides, .. }) => {
                    strides.len() == dimensions.len()
                }
            };
            if !valid_layout {
                return Err(GraphError::InvalidLayout { node: node.id });
            }
            for alias in &node.aliases {
                if usize::from(alias.input_index) >= node.inputs.len() {
                    return Err(GraphError::InvalidAliasInput {
                        node: node.id,
                        input_index: alias.input_index,
                    });
                }
            }
            if let Mutation::WritesInput(input_index) = node.mutation {
                if usize::from(input_index) >= node.inputs.len() {
                    return Err(GraphError::InvalidMutationInput {
                        node: node.id,
                        input_index,
                    });
                }
            }
            let valid_matmul = match (node.kind, node.matmul.as_ref()) {
                (NodeKind::Contraction, Some(contract))
                    if node.inputs.len() >= 2
                        && (node.inputs[0].0 as usize) < index
                        && (node.inputs[1].0 as usize) < index =>
                {
                    let rank = |rank: &Rank| match rank {
                        Rank::Unranked => None,
                        Rank::Ranked(dimensions) => Some(dimensions.len()),
                    };
                    let lhs_rank = rank(&contract.lhs_shape);
                    let rhs_rank = rank(&contract.rhs_shape);
                    let result_rank = rank(&contract.result_shape);
                    contract.lhs_shape == nodes[node.inputs[0].0 as usize].shape.rank
                        && contract.rhs_shape == nodes[node.inputs[1].0 as usize].shape.rank
                        && contract.result_shape == node.shape.rank
                        && !contract.contraction_dimensions.is_empty()
                        && contract.contraction_dimensions.iter().all(|dimension| {
                            lhs_rank.is_none_or(|rank| (dimension.lhs as usize) < rank)
                                && rhs_rank.is_none_or(|rank| (dimension.rhs as usize) < rank)
                        })
                        && contract.batch_dimensions.iter().all(|dimension| {
                            result_rank.is_none_or(|rank| (dimension.result as usize) < rank)
                                && dimension.lhs.is_none_or(|axis| {
                                    lhs_rank.is_none_or(|rank| (axis as usize) < rank)
                                })
                                && dimension.rhs.is_none_or(|axis| {
                                    rhs_rank.is_none_or(|rank| (axis as usize) < rank)
                                })
                        })
                }
                (NodeKind::Contraction, None) => false,
                (_, None) => true,
                (_, Some(_)) => false,
            };
            if !valid_matmul {
                return Err(GraphError::InvalidMatmulContract { node: node.id });
            }
            for input in &node.inputs {
                if input.0 as usize >= index {
                    return Err(GraphError::ForwardReference {
                        node: node.id,
                        input: *input,
                    });
                }
                users[input.0 as usize].push(node.id);
            }
        }
        Ok(Self { nodes, users })
    }

    pub fn nodes(&self) -> &[FusionNode] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &FusionNode {
        &self.nodes[id.0 as usize]
    }

    pub fn users(&self, id: NodeId) -> &[NodeId] {
        &self.users[id.0 as usize]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecializationError {
    TargetMismatch {
        expected: GpuTarget,
        found: GpuTarget,
    },
    UnknownNode(NodeId),
    DuplicateShape(NodeId),
    DuplicateStrides(NodeId),
    MissingShape(NodeId),
    MissingStrides(NodeId),
    RankMismatch {
        node: NodeId,
        expected: usize,
        found: usize,
    },
    DimensionMismatch {
        node: NodeId,
        axis: usize,
        expected: u64,
        found: u64,
    },
    StrideMismatch {
        node: NodeId,
        axis: usize,
        expected: i64,
        found: i64,
    },
    OffsetMismatch {
        node: NodeId,
        expected: i64,
        found: i64,
    },
}

impl fmt::Display for SpecializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for SpecializationError {}

impl KernelSpecialization {
    pub fn validate(
        &self,
        graph: &FusionGraph,
        expected_target: GpuTarget,
    ) -> Result<(), SpecializationError> {
        self.validate_nodes(
            graph,
            expected_target,
            graph.nodes().iter().map(|node| node.id),
        )
    }

    /// Validates a single compiled kernel. Nodes outside the selected region
    /// do not impose runtime-shape requirements on this kernel instance.
    pub fn validate_region(
        &self,
        graph: &FusionGraph,
        region: &FusionRegion,
        expected_target: GpuTarget,
    ) -> Result<(), SpecializationError> {
        self.validate_nodes(
            graph,
            expected_target,
            region.inputs.iter().chain(&region.nodes).copied(),
        )
    }

    fn validate_nodes(
        &self,
        graph: &FusionGraph,
        expected_target: GpuTarget,
        nodes: impl IntoIterator<Item = NodeId>,
    ) -> Result<(), SpecializationError> {
        if self.target != expected_target {
            return Err(SpecializationError::TargetMismatch {
                expected: expected_target,
                found: self.target,
            });
        }
        let mut shapes = BTreeMap::new();
        for shape in &self.shapes {
            if shape.node.0 as usize >= graph.nodes().len() {
                return Err(SpecializationError::UnknownNode(shape.node));
            }
            if shapes
                .insert(shape.node, shape.dimensions.as_slice())
                .is_some()
            {
                return Err(SpecializationError::DuplicateShape(shape.node));
            }
        }
        let mut strides = BTreeMap::new();
        for layout in &self.strides {
            if layout.node.0 as usize >= graph.nodes().len() {
                return Err(SpecializationError::UnknownNode(layout.node));
            }
            if strides.insert(layout.node, layout).is_some() {
                return Err(SpecializationError::DuplicateStrides(layout.node));
            }
        }

        for id in nodes.into_iter().collect::<BTreeSet<_>>() {
            let node = graph.node(id);
            let provided_shape = shapes.get(&node.id).copied();
            let concrete_rank = match &node.shape.rank {
                Rank::Unranked => provided_shape
                    .ok_or(SpecializationError::MissingShape(node.id))?
                    .len(),
                Rank::Ranked(dimensions) => {
                    if let Some(provided) = provided_shape {
                        if dimensions.len() != provided.len() {
                            return Err(SpecializationError::RankMismatch {
                                node: node.id,
                                expected: dimensions.len(),
                                found: provided.len(),
                            });
                        }
                        for (axis, (expected, found)) in dimensions.iter().zip(provided).enumerate()
                        {
                            if let Dimension::Known(expected) = expected {
                                if expected != found {
                                    return Err(SpecializationError::DimensionMismatch {
                                        node: node.id,
                                        axis,
                                        expected: *expected,
                                        found: *found,
                                    });
                                }
                            }
                        }
                    } else if dimensions
                        .iter()
                        .any(|dimension| matches!(dimension, Dimension::Dynamic))
                    {
                        return Err(SpecializationError::MissingShape(node.id));
                    }
                    dimensions.len()
                }
            };

            let provided_strides = strides.get(&node.id).copied();
            let requires_runtime_strides = match &node.layout {
                StorageLayout::Runtime => true,
                StorageLayout::Dense { .. } => false,
                StorageLayout::Strided { strides, offset } => {
                    strides
                        .iter()
                        .any(|stride| matches!(stride, Stride::Dynamic))
                        || matches!(offset, Stride::Dynamic)
                }
            };
            if requires_runtime_strides && provided_strides.is_none() {
                return Err(SpecializationError::MissingStrides(node.id));
            }
            if let Some(provided) = provided_strides {
                if provided.strides.len() != concrete_rank {
                    return Err(SpecializationError::RankMismatch {
                        node: node.id,
                        expected: concrete_rank,
                        found: provided.strides.len(),
                    });
                }
                if let StorageLayout::Strided { strides, offset } = &node.layout {
                    for (axis, (expected, found)) in
                        strides.iter().zip(&provided.strides).enumerate()
                    {
                        if let Stride::Known(expected) = expected {
                            if expected != found {
                                return Err(SpecializationError::StrideMismatch {
                                    node: node.id,
                                    axis,
                                    expected: *expected,
                                    found: *found,
                                });
                            }
                        }
                    }
                    if let Stride::Known(expected) = offset {
                        if *expected != provided.offset {
                            return Err(SpecializationError::OffsetMismatch {
                                node: node.id,
                                expected: *expected,
                                found: provided.offset,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceModel {
    pub memory_bandwidth_bytes_per_second: f64,
    pub peak_flops_per_second: f64,
    pub kernel_launch_overhead_seconds: f64,
    pub shared_memory_per_block: u64,
    pub max_kernel_parameters: usize,
    pub max_unnested_reductions: u16,
    pub max_nodes_per_region: usize,
    pub allow_cheap_duplication: bool,
}

impl DeviceModel {
    pub const fn conservative_gpu() -> Self {
        Self {
            memory_bandwidth_bytes_per_second: 500_000_000_000.0,
            peak_flops_per_second: 10_000_000_000_000.0,
            kernel_launch_overhead_seconds: 5.0e-6,
            shared_memory_per_block: 64 * 1024,
            max_kernel_parameters: 64,
            max_unnested_reductions: 8,
            max_nodes_per_region: 64,
            allow_cheap_duplication: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionHero {
    Reduction,
    Contraction,
    Transpose,
    Gather,
    Scatter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionRejection {
    NotProducerConsumer,
    SideEffects,
    InPlaceAliasing,
    ProducerHasOtherUsers,
    ParameterLimit {
        required: usize,
        limit: usize,
    },
    SharedMemoryBudget {
        required: u64,
        limit: u64,
    },
    ReductionBudget {
        required: u16,
        limit: u16,
    },
    NodeBudget {
        required: usize,
        limit: usize,
    },
    IncompatibleHeroes {
        producer: FusionHero,
        consumer: FusionHero,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusionDecision {
    Allow { benefit_seconds: f64 },
    Forbid(FusionRejection),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionEstimate {
    pub bytes_transferred: u64,
    pub flops: u64,
    pub execution_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionRegion {
    pub id: RegionId,
    pub nodes: Vec<NodeId>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub estimate: RegionEstimate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionPlan {
    pub regions: Vec<FusionRegion>,
    pub node_regions: Vec<Option<RegionId>>,
}

impl FusionPlan {
    pub fn region_for(&self, node: NodeId) -> Option<&FusionRegion> {
        let region = self.node_regions.get(node.0 as usize).copied().flatten()?;
        self.regions.get(region.0 as usize)
    }
}

#[derive(Debug, Clone)]
struct WorkingRegion {
    nodes: BTreeSet<NodeId>,
}

pub fn plan(graph: &FusionGraph, device: DeviceModel) -> FusionPlan {
    let mut regions = graph
        .nodes()
        .iter()
        .filter(|node| !node.kind.is_source())
        .map(|node| {
            (
                node.id,
                WorkingRegion {
                    nodes: BTreeSet::from([node.id]),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut owners = graph
        .nodes()
        .iter()
        .map(|node| (!node.kind.is_source()).then_some(node.id))
        .collect::<Vec<_>>();

    loop {
        let mut best: Option<(f64, NodeId, NodeId)> = None;
        for (&producer_id, producer) in &regions {
            for (&consumer_id, consumer) in &regions {
                if producer_id == consumer_id {
                    continue;
                }
                let FusionDecision::Allow { benefit_seconds } =
                    decide(graph, producer, consumer, device)
                else {
                    continue;
                };
                if benefit_seconds > 0.0
                    && best.is_none_or(|(best_benefit, _, _)| benefit_seconds > best_benefit)
                {
                    best = Some((benefit_seconds, producer_id, consumer_id));
                }
            }
        }
        let Some((_, producer_id, consumer_id)) = best else {
            break;
        };
        let producer = regions
            .remove(&producer_id)
            .expect("candidate still exists");
        let consumer = regions
            .get_mut(&consumer_id)
            .expect("candidate still exists");
        for node in producer.nodes {
            owners[node.0 as usize] = Some(consumer_id);
            consumer.nodes.insert(node);
        }
    }

    let mut ordered = regions.into_values().collect::<Vec<_>>();
    ordered.sort_by_key(|region| region.nodes.iter().next().copied());
    let mut node_regions = vec![None; graph.nodes().len()];
    let mut final_regions = Vec::with_capacity(ordered.len());
    for (index, region) in ordered.into_iter().enumerate() {
        let id = RegionId(index as u32);
        for node in &region.nodes {
            node_regions[node.0 as usize] = Some(id);
        }
        final_regions.push(materialize_region(graph, &region, id, device));
    }
    FusionPlan {
        regions: final_regions,
        node_regions,
    }
}

fn decide(
    graph: &FusionGraph,
    producer: &WorkingRegion,
    consumer: &WorkingRegion,
    device: DeviceModel,
) -> FusionDecision {
    let connected = producer.nodes.iter().any(|node| {
        graph
            .users(*node)
            .iter()
            .any(|user| consumer.nodes.contains(user))
    });
    if !connected {
        return FusionDecision::Forbid(FusionRejection::NotProducerConsumer);
    }
    if producer
        .nodes
        .iter()
        .chain(&consumer.nodes)
        .any(|node| graph.node(*node).has_side_effects)
    {
        return FusionDecision::Forbid(FusionRejection::SideEffects);
    }
    if producer
        .nodes
        .iter()
        .chain(&consumer.nodes)
        .any(|node| !matches!(graph.node(*node).mutation, Mutation::None))
    {
        return FusionDecision::Forbid(FusionRejection::InPlaceAliasing);
    }
    if !device.allow_cheap_duplication
        && producer.nodes.iter().any(|node| {
            graph
                .users(*node)
                .iter()
                .any(|user| !consumer.nodes.contains(user) && !producer.nodes.contains(user))
        })
    {
        return FusionDecision::Forbid(FusionRejection::ProducerHasOtherUsers);
    }
    let combined = WorkingRegion {
        nodes: producer.nodes.union(&consumer.nodes).copied().collect(),
    };
    if combined.nodes.len() > device.max_nodes_per_region {
        return FusionDecision::Forbid(FusionRejection::NodeBudget {
            required: combined.nodes.len(),
            limit: device.max_nodes_per_region,
        });
    }
    let parameters =
        external_inputs(graph, &combined).len() + external_outputs(graph, &combined).len();
    if parameters > device.max_kernel_parameters {
        return FusionDecision::Forbid(FusionRejection::ParameterLimit {
            required: parameters,
            limit: device.max_kernel_parameters,
        });
    }
    let shared_memory = combined
        .nodes
        .iter()
        .map(|node| graph.node(*node).shared_memory_bytes)
        .sum::<u64>();
    if shared_memory > device.shared_memory_per_block {
        return FusionDecision::Forbid(FusionRejection::SharedMemoryBudget {
            required: shared_memory,
            limit: device.shared_memory_per_block,
        });
    }
    let reductions = combined
        .nodes
        .iter()
        .map(|node| graph.node(*node).unnested_reductions)
        .sum::<u16>();
    if reductions > device.max_unnested_reductions {
        return FusionDecision::Forbid(FusionRejection::ReductionBudget {
            required: reductions,
            limit: device.max_unnested_reductions,
        });
    }
    if let (Some(producer), Some(consumer)) = (hero(graph, producer), hero(graph, consumer)) {
        if producer != consumer {
            return FusionDecision::Forbid(FusionRejection::IncompatibleHeroes {
                producer,
                consumer,
            });
        }
    }
    let unfused = estimate(graph, producer, device).execution_seconds
        + estimate(graph, consumer, device).execution_seconds;
    let fused = estimate(graph, &combined, device).execution_seconds;
    FusionDecision::Allow {
        benefit_seconds: unfused - fused,
    }
}

fn hero(graph: &FusionGraph, region: &WorkingRegion) -> Option<FusionHero> {
    region
        .nodes
        .iter()
        .rev()
        .find_map(|node| graph.node(*node).kind.hero())
}

fn estimate(graph: &FusionGraph, region: &WorkingRegion, device: DeviceModel) -> RegionEstimate {
    let raw_bytes = region
        .nodes
        .iter()
        .map(|node| {
            let node = graph.node(*node);
            node.bytes_read.saturating_add(node.bytes_written)
        })
        .sum::<u64>();
    let internal_bytes = region
        .nodes
        .iter()
        .flat_map(|node| {
            graph
                .users(*node)
                .iter()
                .filter(|user| region.nodes.contains(user))
                .map(|_| graph.node(*node).shape.byte_size().unwrap_or(0))
        })
        .sum::<u64>();
    let bytes_transferred = raw_bytes.saturating_sub(internal_bytes.saturating_mul(2));
    let flops = region
        .nodes
        .iter()
        .map(|node| graph.node(*node).flops)
        .sum::<u64>();
    let memory_seconds = bytes_transferred as f64 / device.memory_bandwidth_bytes_per_second;
    let compute_seconds = flops as f64 / device.peak_flops_per_second;
    RegionEstimate {
        bytes_transferred,
        flops,
        execution_seconds: device.kernel_launch_overhead_seconds
            + memory_seconds.max(compute_seconds),
    }
}

fn external_inputs(graph: &FusionGraph, region: &WorkingRegion) -> Vec<NodeId> {
    region
        .nodes
        .iter()
        .flat_map(|node| &graph.node(*node).inputs)
        .filter(|input| !region.nodes.contains(input))
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn external_outputs(graph: &FusionGraph, region: &WorkingRegion) -> Vec<NodeId> {
    region
        .nodes
        .iter()
        .filter(|node| {
            graph.users(**node).is_empty()
                || graph
                    .users(**node)
                    .iter()
                    .any(|user| !region.nodes.contains(user))
        })
        .copied()
        .collect()
}

fn materialize_region(
    graph: &FusionGraph,
    region: &WorkingRegion,
    id: RegionId,
    device: DeviceModel,
) -> FusionRegion {
    FusionRegion {
        id,
        nodes: region.nodes.iter().copied().collect(),
        inputs: external_inputs(graph, region),
        outputs: external_outputs(graph, region),
        estimate: estimate(graph, region, device),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, kind: NodeKind, inputs: &[u32]) -> FusionNode {
        let mut node = FusionNode::structural(
            id,
            kind,
            inputs.iter().copied().map(NodeId),
            Shape::ranked([2, 4], 32),
        );
        node.flops = 8;
        node
    }

    #[test]
    fn rms_norm_graph_becomes_one_profitable_region() {
        let graph = FusionGraph::new(vec![
            node(0, NodeKind::Parameter, &[]),
            node(1, NodeKind::Parameter, &[]),
            node(2, NodeKind::Elementwise, &[0, 0]),
            node(3, NodeKind::Reduction, &[2]),
            node(4, NodeKind::Elementwise, &[3]),
            node(5, NodeKind::Elementwise, &[4]),
            node(6, NodeKind::Elementwise, &[0, 5]),
            node(7, NodeKind::Elementwise, &[1, 6]),
        ])
        .unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        assert_eq!(plan.regions.len(), 1);
        assert_eq!(
            plan.regions[0].nodes,
            (2..=7).map(NodeId).collect::<Vec<_>>()
        );
        assert_eq!(plan.regions[0].inputs, [NodeId(0), NodeId(1)]);
        assert_eq!(plan.regions[0].outputs, [NodeId(7)]);
    }

    #[test]
    fn full_graph_mapping_survives_unfusible_side_effects() {
        let mut effect = node(2, NodeKind::Scatter, &[1]);
        effect.has_side_effects = true;
        let graph = FusionGraph::new(vec![
            node(0, NodeKind::Parameter, &[]),
            node(1, NodeKind::Elementwise, &[0]),
            effect,
            node(3, NodeKind::Elementwise, &[2]),
        ])
        .unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        assert_eq!(plan.regions.len(), 3);
        assert!(plan.region_for(NodeId(0)).is_none());
        assert!(plan.region_for(NodeId(1)).is_some());
        assert!(plan.region_for(NodeId(2)).is_some());
        assert!(plan.region_for(NodeId(3)).is_some());
    }

    #[test]
    fn shared_memory_and_reduction_budgets_are_hard_limits() {
        let mut reduction = node(1, NodeKind::Reduction, &[0]);
        reduction.shared_memory_bytes = 65 * 1024;
        reduction.unnested_reductions = 9;
        let graph = FusionGraph::new(vec![
            node(0, NodeKind::Parameter, &[]),
            reduction,
            node(2, NodeKind::Elementwise, &[1]),
        ])
        .unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        assert_eq!(plan.regions.len(), 2);
    }

    #[test]
    fn a_producer_with_an_external_user_is_not_silently_duplicated() {
        let graph = FusionGraph::new(vec![
            node(0, NodeKind::Parameter, &[]),
            node(1, NodeKind::Elementwise, &[0]),
            node(2, NodeKind::Elementwise, &[1]),
            node(3, NodeKind::Reduction, &[1]),
        ])
        .unwrap();
        let plan = plan(&graph, DeviceModel::conservative_gpu());
        assert_eq!(plan.regions.len(), 3);
        assert_ne!(plan.node_regions[2], plan.node_regions[3]);
    }

    #[test]
    fn rank_zero_and_unranked_are_distinct_graph_types() {
        let scalar = Shape::typed([], ElementKind::IeeeFloat, 32);
        let unranked = Shape::unranked(ElementKind::IeeeFloat, 32);
        assert_eq!(scalar.rank, Rank::Ranked(Vec::new()));
        assert_eq!(unranked.rank, Rank::Unranked);
        assert_ne!(scalar, unranked);
    }

    #[test]
    fn specialization_resolves_dynamic_shape_stride_and_target_data() {
        let mut parameter = FusionNode::structural(
            0,
            NodeKind::Parameter,
            [],
            Shape::typed(
                [Dimension::Dynamic, Dimension::Known(4)],
                ElementKind::BrainFloat,
                16,
            ),
        );
        parameter.layout = StorageLayout::Strided {
            strides: vec![Stride::Known(4), Stride::Dynamic],
            offset: Stride::Known(2),
        };
        let graph = FusionGraph::new(vec![parameter]).unwrap();
        let specialization = KernelSpecialization {
            shapes: vec![RuntimeShape {
                node: NodeId(0),
                dimensions: vec![3, 4],
            }],
            strides: vec![RuntimeStrides {
                node: NodeId(0),
                strides: vec![4, 1],
                offset: 2,
            }],
            target: GpuTarget::Amd,
        };
        assert_eq!(specialization.validate(&graph, GpuTarget::Amd), Ok(()));
        assert!(matches!(
            specialization.validate(&graph, GpuTarget::Nvidia),
            Err(SpecializationError::TargetMismatch { .. })
        ));

        let mut wrong_stride = specialization.clone();
        wrong_stride.strides[0].strides[0] = 8;
        assert!(matches!(
            wrong_stride.validate(&graph, GpuTarget::Amd),
            Err(SpecializationError::StrideMismatch { axis: 0, .. })
        ));
    }

    #[test]
    fn region_specialization_does_not_require_unrelated_graph_nodes() {
        let mut dynamic_input = node(0, NodeKind::Parameter, &[]);
        dynamic_input.shape = Shape::unranked(ElementKind::IeeeFloat, 32);
        dynamic_input.layout = StorageLayout::Runtime;
        let mut consumer = node(1, NodeKind::Elementwise, &[0]);
        consumer.shape = Shape::typed(
            [Dimension::Dynamic, Dimension::Known(4)],
            ElementKind::IeeeFloat,
            32,
        );
        consumer.layout = consumer.shape.default_layout();
        let mut unrelated = node(2, NodeKind::Parameter, &[]);
        unrelated.shape = Shape::unranked(ElementKind::IeeeFloat, 32);
        unrelated.layout = StorageLayout::Runtime;
        let graph = FusionGraph::new(vec![dynamic_input, consumer, unrelated]).unwrap();
        let fusion_plan = plan(&graph, DeviceModel::conservative_gpu());
        let region = fusion_plan
            .regions
            .iter()
            .find(|region| region.nodes.contains(&NodeId(1)))
            .unwrap();
        let specialization = KernelSpecialization {
            shapes: vec![
                RuntimeShape {
                    node: NodeId(0),
                    dimensions: vec![2, 4],
                },
                RuntimeShape {
                    node: NodeId(1),
                    dimensions: vec![2, 4],
                },
            ],
            strides: vec![RuntimeStrides {
                node: NodeId(0),
                strides: vec![4, 1],
                offset: 0,
            }],
            target: GpuTarget::Amd,
        };
        assert_eq!(
            specialization.validate_region(&graph, region, GpuTarget::Amd),
            Ok(())
        );
        assert_eq!(
            specialization.validate(&graph, GpuTarget::Amd),
            Err(SpecializationError::MissingShape(NodeId(2)))
        );
    }
}
