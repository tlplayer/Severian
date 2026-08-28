use severian_artifact::{ArtifactId, CompiledRegionId};
use severian_mir::{
    Block as MirBlock, Function as MirFunction, Module as MirModule, Operation as MirOperation,
    Value as MirValue,
};
use severian_target::TargetSpec;
use severian_universal::{
    tensor, Attrs, CompilerId, ExecutionPlacement, FloatFormat, IntegerWidth, OpId,
    PrimitiveRepresentation, TypeContext, TypeId,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectSet {
    pub reads_memory: bool,
    pub writes_memory: bool,
    pub may_trap: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileRegion {
    pub id: CompiledRegionId,
    pub compiler: CompilerId,
    pub operations: Vec<MirOperation>,
    pub compile_operations: Vec<CompileOperation>,
    /// Value slots returned by the region entry point. Input slots are always
    /// `0..inputs.len()`; operation result slots follow them.
    pub output_slots: Vec<u32>,
    pub inputs: Vec<MirValue>,
    pub outputs: Vec<MirValue>,
    /// Unspecialized physical and logical contracts for every region-local
    /// value slot. Runtime specialization refines these records; emitters do
    /// not recover them from native pointers.
    pub value_contracts: Vec<CompileValueContract>,
    pub effects: EffectSet,
    /// Source execution intent for the complete region. Backend selection is
    /// performed before invoking a target-specific emitter.
    pub placement: Option<ExecutionPlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileValueContract {
    pub slot: u32,
    pub type_id: TypeId,
    pub tensor: Option<TensorValueContract>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorValueContract {
    pub element_kind: severian_fusion::ElementKind,
    pub element_bits: u16,
    pub rank: severian_fusion::Rank,
    pub layout: severian_fusion::StorageLayout,
    pub aliases: Vec<ValueAlias>,
    pub mutation: ValueMutation,
    pub runtime_shape_operands: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueAlias {
    pub source_slot: u32,
    pub kind: severian_fusion::AliasKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueMutation {
    #[default]
    None,
    WritesSlot(u32),
}

/// Concrete runtime facts used to refine one region-local tensor value before
/// rank-dependent backend emission. The slot is structural identity; dtype and
/// rank remain ordinary fields rather than operation or symbol names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeValueSpecialization {
    pub slot: u32,
    pub dimensions: Vec<u64>,
    pub strides: Vec<i64>,
    pub offset: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompileRegionSpecialization {
    pub values: Vec<RuntimeValueSpecialization>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionSpecializationError {
    DuplicateValue(u32),
    UnknownValue(u32),
    MissingRuntimeShape(u32),
    RankMismatch {
        slot: u32,
        expected: usize,
        found: usize,
    },
    DimensionMismatch {
        slot: u32,
        axis: usize,
        expected: u64,
        found: u64,
    },
    StrideRankMismatch {
        slot: u32,
        rank: usize,
        strides: usize,
    },
    StrideMismatch {
        slot: u32,
        axis: usize,
        expected: i64,
        found: i64,
    },
    OffsetMismatch {
        slot: u32,
        expected: i64,
        found: i64,
    },
    InvalidType(String),
}

impl std::fmt::Display for RegionSpecializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RegionSpecializationError {}

impl CompileRegion {
    pub fn rebuild_value_contracts(&mut self, types: &TypeContext) -> Result<(), String> {
        let mut contracts = BTreeMap::new();
        for (slot, value) in self.inputs.iter().enumerate() {
            contracts.insert(
                slot as u32,
                compile_value_contract(slot as u32, value.type_id, types, true)?,
            );
        }
        let mut next_implicit_slot = self.inputs.len() as u32;
        for operation in &self.compile_operations {
            let structural = tensor::TensorOp::decode(operation.id, &operation.attributes);
            let operand_slots =
                if operation.operand_slots.is_empty() && self.compile_operations.len() == 1 {
                    (0..operation.operands.len() as u32).collect::<Vec<_>>()
                } else {
                    operation.operand_slots.clone()
                };
            let result_slots =
                if operation.result_slots.is_empty() && self.compile_operations.len() == 1 {
                    let slots = (next_implicit_slot
                        ..next_implicit_slot + operation.results.len() as u32)
                        .collect::<Vec<_>>();
                    next_implicit_slot += operation.results.len() as u32;
                    slots
                } else {
                    operation.result_slots.clone()
                };
            let runtime_shape_operands = structural
                .map(tensor_runtime_shape_operand_indices)
                .unwrap_or_default()
                .iter()
                .filter_map(|index| operand_slots.get(*index).copied())
                .collect::<Vec<_>>();
            for (slot, type_id) in result_slots.into_iter().zip(&operation.results) {
                let mut contract = compile_value_contract(slot, *type_id, types, false)?;
                if let Some(tensor) = &mut contract.tensor {
                    tensor.runtime_shape_operands = runtime_shape_operands.clone();
                    match structural {
                        Some(tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Reshape))
                        | Some(tensor::TensorOp::Slice) => {
                            if let Some(source_slot) = operand_slots.first() {
                                tensor.aliases.push(ValueAlias {
                                    source_slot: *source_slot,
                                    kind: severian_fusion::AliasKind::View,
                                });
                            }
                            tensor.layout = match &tensor.rank {
                                severian_fusion::Rank::Ranked(dimensions)
                                    if structural == Some(tensor::TensorOp::Slice) =>
                                {
                                    severian_fusion::StorageLayout::Strided {
                                        strides: vec![
                                            severian_fusion::Stride::Dynamic;
                                            dimensions.len()
                                        ],
                                        offset: severian_fusion::Stride::Dynamic,
                                    }
                                }
                                _ => severian_fusion::StorageLayout::Runtime,
                            };
                        }
                        Some(tensor::TensorOp::Scatter) => {
                            if let Some(source_slot) = operand_slots.first() {
                                tensor.aliases.push(ValueAlias {
                                    source_slot: *source_slot,
                                    kind: severian_fusion::AliasKind::InPlace,
                                });
                                tensor.mutation = ValueMutation::WritesSlot(*source_slot);
                                if let Some(source) = contracts.get(source_slot).and_then(
                                    |contract: &CompileValueContract| contract.tensor.as_ref(),
                                ) {
                                    tensor.layout = source.layout.clone();
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if contracts.insert(slot, contract).is_some() {
                    return Err(format!("region value slot {slot} is defined twice"));
                }
            }
        }
        self.value_contracts = contracts.into_values().collect();
        Ok(())
    }

    /// Produces the ranked region and matching type context consumed by CPU
    /// MLIR emission. Runtime dimensions validate dynamic/static constraints,
    /// but only previously-unranked tensors have their rank refined; ranked
    /// dynamic dimensions stay dynamic in the emitted ABI.
    pub fn specialize_for_emission(
        &self,
        types: &TypeContext,
        specialization: &CompileRegionSpecialization,
    ) -> Result<(Self, TypeContext), RegionSpecializationError> {
        let mut region = self.clone();
        if region.value_contracts.is_empty() {
            region
                .rebuild_value_contracts(types)
                .map_err(RegionSpecializationError::InvalidType)?;
        }
        let contracts = region
            .value_contracts
            .iter()
            .map(|contract| (contract.slot, contract.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut runtime_values = BTreeMap::new();
        for value in &specialization.values {
            if !contracts.contains_key(&value.slot) {
                return Err(RegionSpecializationError::UnknownValue(value.slot));
            }
            if runtime_values.insert(value.slot, value).is_some() {
                return Err(RegionSpecializationError::DuplicateValue(value.slot));
            }
        }
        let required = rank_dependent_slots(&region);
        let mut specialized_types = TypeContext::clone(types);
        let mut slot_types = BTreeMap::new();
        for contract in &mut region.value_contracts {
            let Some(tensor) = &mut contract.tensor else {
                slot_types.insert(contract.slot, contract.type_id);
                continue;
            };
            let runtime = runtime_values.get(&contract.slot).copied();
            if matches!(tensor.rank, severian_fusion::Rank::Unranked)
                && required.contains(&contract.slot)
                && runtime.is_none()
            {
                return Err(RegionSpecializationError::MissingRuntimeShape(
                    contract.slot,
                ));
            }
            let Some(runtime) = runtime else {
                slot_types.insert(contract.slot, contract.type_id);
                continue;
            };
            validate_runtime_value(contract.slot, tensor, runtime)?;
            let refined_rank = match &tensor.rank {
                severian_fusion::Rank::Unranked => severian_fusion::Rank::Ranked(vec![
                        severian_fusion::Dimension::Dynamic;
                        runtime.dimensions.len()
                    ]),
                ranked => ranked.clone(),
            };
            if tensor.rank != refined_rank {
                let shape = fusion_rank_to_tensor_shape(&refined_rank);
                contract.type_id = specialized_types
                    .refine_tensor_shape(contract.type_id, shape)
                    .map_err(|error| RegionSpecializationError::InvalidType(error.to_string()))?;
                tensor.rank = refined_rank;
            }
            tensor.layout = severian_fusion::StorageLayout::Strided {
                strides: runtime
                    .strides
                    .iter()
                    .copied()
                    .map(severian_fusion::Stride::Known)
                    .collect(),
                offset: severian_fusion::Stride::Known(runtime.offset),
            };
            slot_types.insert(contract.slot, contract.type_id);
        }

        rewrite_region_types(&mut region, &slot_types)?;
        Ok((region, specialized_types))
    }
}

fn validate_runtime_value(
    slot: u32,
    contract: &TensorValueContract,
    runtime: &RuntimeValueSpecialization,
) -> Result<(), RegionSpecializationError> {
    if runtime.dimensions.len() != runtime.strides.len() {
        return Err(RegionSpecializationError::StrideRankMismatch {
            slot,
            rank: runtime.dimensions.len(),
            strides: runtime.strides.len(),
        });
    }
    if let severian_fusion::Rank::Ranked(dimensions) = &contract.rank {
        if dimensions.len() != runtime.dimensions.len() {
            return Err(RegionSpecializationError::RankMismatch {
                slot,
                expected: dimensions.len(),
                found: runtime.dimensions.len(),
            });
        }
        for (axis, (expected, found)) in dimensions.iter().zip(&runtime.dimensions).enumerate() {
            if let severian_fusion::Dimension::Known(expected) = expected {
                if expected != found {
                    return Err(RegionSpecializationError::DimensionMismatch {
                        slot,
                        axis,
                        expected: *expected,
                        found: *found,
                    });
                }
            }
        }
    }
    if let severian_fusion::StorageLayout::Strided { strides, offset } = &contract.layout {
        if strides.len() != runtime.strides.len() {
            return Err(RegionSpecializationError::StrideRankMismatch {
                slot,
                rank: strides.len(),
                strides: runtime.strides.len(),
            });
        }
        for (axis, (expected, found)) in strides.iter().zip(&runtime.strides).enumerate() {
            if let severian_fusion::Stride::Known(expected) = expected {
                if expected != found {
                    return Err(RegionSpecializationError::StrideMismatch {
                        slot,
                        axis,
                        expected: *expected,
                        found: *found,
                    });
                }
            }
        }
        if let severian_fusion::Stride::Known(expected) = offset {
            if *expected != runtime.offset {
                return Err(RegionSpecializationError::OffsetMismatch {
                    slot,
                    expected: *expected,
                    found: runtime.offset,
                });
            }
        }
    }
    Ok(())
}

fn fusion_rank_to_tensor_shape(rank: &severian_fusion::Rank) -> severian_universal::TensorShape {
    match rank {
        severian_fusion::Rank::Unranked => severian_universal::TensorShape::Unranked,
        severian_fusion::Rank::Ranked(dimensions) => severian_universal::TensorShape::Ranked(
            dimensions
                .iter()
                .map(|dimension| match dimension {
                    severian_fusion::Dimension::Dynamic => {
                        severian_universal::TensorDimension::Dynamic
                    }
                    severian_fusion::Dimension::Known(value) => {
                        severian_universal::TensorDimension::Known(*value)
                    }
                })
                .collect(),
        ),
    }
}

fn rank_dependent_slots(region: &CompileRegion) -> std::collections::BTreeSet<u32> {
    let mut required = std::collections::BTreeSet::new();
    let mut next_implicit_slot = region.inputs.len() as u32;
    for operation in &region.compile_operations {
        let Some(structural) = tensor::TensorOp::decode(operation.id, &operation.attributes) else {
            continue;
        };
        if structural == tensor::TensorOp::StorageView(tensor::StorageViewOp::Shape) {
            continue;
        }
        let operands = operation_operand_slots(region, operation);
        required.extend(operands);
        let results = operation_result_slots(region, operation, &mut next_implicit_slot);
        required.extend(results);
    }
    required
}

fn rewrite_region_types(
    region: &mut CompileRegion,
    slot_types: &BTreeMap<u32, TypeId>,
) -> Result<(), RegionSpecializationError> {
    for (slot, value) in region.inputs.iter_mut().enumerate() {
        if let Some(type_id) = slot_types.get(&(slot as u32)) {
            value.type_id = *type_id;
        }
    }
    let mut next_implicit_slot = region.inputs.len() as u32;
    let single_operation = region.compile_operations.len() == 1;
    for operation in &mut region.compile_operations {
        let operand_slots = if operation.operand_slots.is_empty() && single_operation {
            (0..operation.operands.len() as u32).collect::<Vec<_>>()
        } else {
            operation.operand_slots.clone()
        };
        let result_slots = if operation.result_slots.is_empty() && single_operation {
            let slots = (next_implicit_slot..next_implicit_slot + operation.results.len() as u32)
                .collect::<Vec<_>>();
            next_implicit_slot += operation.results.len() as u32;
            slots
        } else {
            operation.result_slots.clone()
        };
        for (slot, type_id) in operand_slots.into_iter().zip(&mut operation.operands) {
            *type_id = *slot_types
                .get(&slot)
                .ok_or(RegionSpecializationError::UnknownValue(slot))?;
        }
        for (slot, type_id) in result_slots.into_iter().zip(&mut operation.results) {
            *type_id = *slot_types
                .get(&slot)
                .ok_or(RegionSpecializationError::UnknownValue(slot))?;
        }
    }
    let output_slots = resolved_output_slots(region);
    for (slot, value) in output_slots.into_iter().zip(&mut region.outputs) {
        value.type_id = *slot_types
            .get(&slot)
            .ok_or(RegionSpecializationError::UnknownValue(slot))?;
    }
    Ok(())
}

fn operation_operand_slots(region: &CompileRegion, operation: &CompileOperation) -> Vec<u32> {
    if operation.operand_slots.is_empty() && region.compile_operations.len() == 1 {
        (0..operation.operands.len() as u32).collect()
    } else {
        operation.operand_slots.clone()
    }
}

fn operation_result_slots(
    region: &CompileRegion,
    operation: &CompileOperation,
    next_implicit_slot: &mut u32,
) -> Vec<u32> {
    if operation.result_slots.is_empty() && region.compile_operations.len() == 1 {
        let slots =
            (*next_implicit_slot..*next_implicit_slot + operation.results.len() as u32).collect();
        *next_implicit_slot += operation.results.len() as u32;
        slots
    } else {
        operation.result_slots.clone()
    }
}

fn resolved_output_slots(region: &CompileRegion) -> Vec<u32> {
    if !region.output_slots.is_empty() {
        return region.output_slots.clone();
    }
    let mut next_implicit_slot = region.inputs.len() as u32;
    let mut outputs = Vec::new();
    for operation in &region.compile_operations {
        outputs = operation_result_slots(region, operation, &mut next_implicit_slot);
    }
    outputs
}

fn compile_value_contract(
    slot: u32,
    type_id: TypeId,
    types: &TypeContext,
    external: bool,
) -> Result<CompileValueContract, String> {
    let tensor = types
        .tensor(type_id)
        .map(|tensor_type| -> Result<TensorValueContract, String> {
            let (element_kind, element_bits) = tensor_element_contract(types, tensor_type.element)?;
            let rank = match tensor_type.shape {
                severian_universal::TensorShape::Unranked => severian_fusion::Rank::Unranked,
                severian_universal::TensorShape::Ranked(dimensions) => {
                    severian_fusion::Rank::Ranked(
                        dimensions
                            .into_iter()
                            .map(|dimension| match dimension {
                                severian_universal::TensorDimension::Dynamic => {
                                    severian_fusion::Dimension::Dynamic
                                }
                                severian_universal::TensorDimension::Known(value) => {
                                    severian_fusion::Dimension::Known(value)
                                }
                            })
                            .collect(),
                    )
                }
            };
            let layout = if external || matches!(rank, severian_fusion::Rank::Unranked) {
                severian_fusion::StorageLayout::Runtime
            } else {
                let severian_fusion::Rank::Ranked(dimensions) = &rank else {
                    unreachable!()
                };
                severian_fusion::StorageLayout::Dense {
                    minor_to_major: (0..dimensions.len() as u32).rev().collect(),
                }
            };
            Ok(TensorValueContract {
                element_kind,
                element_bits,
                rank,
                layout,
                aliases: Vec::new(),
                mutation: ValueMutation::None,
                runtime_shape_operands: Vec::new(),
            })
        })
        .transpose()?;
    Ok(CompileValueContract {
        slot,
        type_id,
        tensor,
    })
}

fn tensor_element_contract(
    types: &TypeContext,
    element: TypeId,
) -> Result<(severian_fusion::ElementKind, u16), String> {
    let representation = types
        .primitive(element)
        .ok_or_else(|| format!("tensor element {element:?} has no physical representation"))?
        .representation;
    match representation {
        PrimitiveRepresentation::Integer {
            bits: IntegerWidth::Fixed(bits),
            signed: true,
        } => Ok((severian_fusion::ElementKind::SignedInteger, bits)),
        PrimitiveRepresentation::Integer {
            bits: IntegerWidth::Fixed(bits),
            signed: false,
        } => Ok((severian_fusion::ElementKind::UnsignedInteger, bits)),
        PrimitiveRepresentation::Float {
            format: FloatFormat::Float8E4M3Fn,
        } => Ok((severian_fusion::ElementKind::Float8E4M3Fn, 8)),
        PrimitiveRepresentation::Float {
            format: FloatFormat::Float8E5M2,
        } => Ok((severian_fusion::ElementKind::Float8E5M2, 8)),
        PrimitiveRepresentation::Float {
            format: FloatFormat::Ieee(bits),
        } => Ok((severian_fusion::ElementKind::IeeeFloat, bits)),
        PrimitiveRepresentation::Float {
            format: FloatFormat::BrainFloat16,
        } => Ok((severian_fusion::ElementKind::BrainFloat, 16)),
        PrimitiveRepresentation::Boolean => Ok((severian_fusion::ElementKind::Boolean, 1)),
        _ => Err(format!(
            "tensor element {element:?} has unsupported representation {representation:?}"
        )),
    }
}

fn tensor_runtime_shape_operand_indices(operation: tensor::TensorOp) -> &'static [usize] {
    match operation {
        tensor::TensorOp::Reduce(tensor::ReductionOp::SumAxis) => &[],
        tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Reshape)
        | tensor::TensorOp::Permute(tensor::PermuteOp::Axes) => &[1],
        tensor::TensorOp::Slice => &[1, 2, 3],
        tensor::TensorOp::Broadcast(_) => &[1],
        tensor::TensorOp::Concatenate => &[2],
        tensor::TensorOp::StorageView(tensor::StorageViewOp::FromElements) => &[1],
        _ => &[],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOperation {
    pub id: OpId,
    pub operands: Vec<TypeId>,
    pub results: Vec<TypeId>,
    /// Region-local SSA slots corresponding one-for-one with `operands` and
    /// `results`. These make data flow explicit across operations in a region.
    pub operand_slots: Vec<u32>,
    pub result_slots: Vec<u32>,
    pub attributes: Attrs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandardRegion {
    pub operations: Vec<MirOperation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanSegment {
    Standard(StandardRegion),
    Compiler(CompileRegion),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlannedBlock {
    pub segments: Vec<PlanSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedFunction {
    pub declaration: MirFunction,
    pub body: Option<PlannedBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilePlan {
    pub source: MirModule,
    pub initializer: PlannedBlock,
    pub functions: Vec<PlannedFunction>,
    /// Custom regions extracted from nested standard control-flow blocks.
    pub nested_regions: Vec<CompileRegion>,
}

impl CompilePlan {
    pub fn has_custom_regions(&self) -> bool {
        !self.nested_regions.is_empty()
            || self
                .initializer
                .segments
                .iter()
                .chain(
                    self.functions
                        .iter()
                        .filter_map(|function| function.body.as_ref())
                        .flat_map(|body| &body.segments),
                )
                .any(|segment| matches!(segment, PlanSegment::Compiler(_)))
    }

    /// Replaces every custom region with a typed generated-function call. The
    /// generic lowerer therefore never observes custom region operations.
    pub fn resumed_mir(&self) -> MirModule {
        self.source.clone()
    }
}

#[allow(dead_code)]
pub(crate) fn resume_block(block: &PlannedBlock) -> MirBlock {
    let mut operations = Vec::new();
    for segment in &block.segments {
        match segment {
            PlanSegment::Standard(region) => operations.extend(region.operations.iter().cloned()),
            PlanSegment::Compiler(region) => {
                operations.push(MirOperation::CompiledRegionCall {
                    artifact: ArtifactId::for_region(region.id),
                    inputs: region.inputs.iter().map(|value| value.id).collect(),
                    outputs: region.outputs.iter().map(|value| value.id).collect(),
                });
            }
        }
    }
    MirBlock { operations }
}

#[derive(Debug, Clone, Copy)]
pub struct CompileContext<'a> {
    pub types: &'a TypeContext,
    pub target: &'a TargetSpec,
}

/// A device-neutral compiler product for a tensor GPU region. It deliberately
/// retains the complete Severian graph and fusion decisions; TTIR and target
/// code are later phases of the Triton bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuKernelBundle {
    pub target: severian_fusion::GpuTarget,
    pub architecture: String,
    pub graph: severian_fusion::FusionGraph,
    pub plan: severian_fusion::FusionPlan,
    /// CompileRegion value slot to fusion-graph node. StorageView aliases and
    /// multi-operation regions make this mapping non-positional.
    pub value_nodes: BTreeMap<u32, severian_fusion::NodeId>,
    pub inputs: Vec<severian_mlir::LoweredType>,
    pub outputs: Vec<severian_mlir::LoweredType>,
}

impl GpuKernelBundle {
    pub fn validate_specialization(
        &self,
        specialization: &severian_fusion::KernelSpecialization,
    ) -> Result<(), severian_fusion::SpecializationError> {
        specialization.validate(&self.graph, self.target)
    }

    pub fn compile_region_specialization(
        &self,
        specialization: &severian_fusion::KernelSpecialization,
    ) -> Result<CompileRegionSpecialization, RegionSpecializationError> {
        let shapes = specialization
            .shapes
            .iter()
            .map(|shape| (shape.node, shape.dimensions.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let strides = specialization
            .strides
            .iter()
            .map(|strides| (strides.node, strides))
            .collect::<BTreeMap<_, _>>();
        let mut values = Vec::with_capacity(self.value_nodes.len());
        for (slot, node) in &self.value_nodes {
            let dimensions = shapes
                .get(node)
                .ok_or(RegionSpecializationError::MissingRuntimeShape(*slot))?;
            let layout = strides
                .get(node)
                .ok_or(RegionSpecializationError::MissingRuntimeShape(*slot))?;
            values.push(RuntimeValueSpecialization {
                slot: *slot,
                dimensions: dimensions.to_vec(),
                strides: layout.strides.clone(),
                offset: layout.offset,
            });
        }
        Ok(CompileRegionSpecialization { values })
    }
}

/// Raw custom-compiler result. GPU regions are not represented as MLIR
/// functions and therefore cannot accidentally enter CPU artifact verification.
#[derive(Debug, Clone, PartialEq)]
pub enum CompiledRegionArtifact {
    CpuMlir(severian_mlir::MlirArtifact),
    GpuKernel(GpuKernelBundle),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedGpuKernelBundle {
    pub id: ArtifactId,
    pub bundle: GpuKernelBundle,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerifiedCompiledRegionArtifact {
    CpuMlir(severian_mlir::VerifiedMlirArtifact),
    GpuKernel(VerifiedGpuKernelBundle),
}
