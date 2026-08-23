use severian_artifact::{ArtifactId, CompiledRegionId};
use severian_mir::{
    Block as MirBlock, Function as MirFunction, Module as MirModule, Operation as MirOperation,
    Value as MirValue,
};
use severian_target::TargetSpec;
use severian_universal::{CompilerId, TypeContext};

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
    pub inputs: Vec<MirValue>,
    pub outputs: Vec<MirValue>,
    pub effects: EffectSet,
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
