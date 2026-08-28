use severian_artifact::{ArtifactId, CompiledRegionId};
use severian_mir::{
    Block as MirBlock, Function as MirFunction, Module as MirModule, Operation as MirOperation,
    Value as MirValue,
};
use severian_target::TargetSpec;
use severian_universal::{Attrs, CompilerId, ExecutionPlacement, OpId, TypeContext, TypeId};

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
    pub effects: EffectSet,
    /// Source execution intent for the complete region. Backend selection is
    /// performed before invoking a target-specific emitter.
    pub placement: Option<ExecutionPlacement>,
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
