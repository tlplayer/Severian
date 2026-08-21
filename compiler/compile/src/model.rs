use severian_artifact::{ArtifactId, CompiledRegionId};
use severian_mir::{Module as MirModule, Operation as MirOperation, Value as MirValue};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilePlan {
    pub source: MirModule,
    pub segments: Vec<PlanSegment>,
}

impl CompilePlan {
    pub fn has_custom_regions(&self) -> bool {
        self.segments
            .iter()
            .any(|segment| matches!(segment, PlanSegment::Compiler(_)))
    }

    /// Replaces every custom region with a typed generated-function call. The
    /// generic lowerer therefore never observes custom region operations.
    pub fn resumed_mir(&self) -> MirModule {
        let mut module = MirModule {
            values: self.source.values.clone(),
            bindings: self.source.bindings.clone(),
            operations: Vec::new(),
        };
        for segment in &self.segments {
            match segment {
                PlanSegment::Standard(region) => {
                    module.operations.extend(region.operations.iter().cloned());
                }
                PlanSegment::Compiler(region) => {
                    module.operations.push(MirOperation::CompiledRegionCall {
                        artifact: ArtifactId::for_region(region.id),
                        inputs: region.inputs.iter().map(|value| value.id).collect(),
                        outputs: region.outputs.iter().map(|value| value.id).collect(),
                    });
                }
            }
        }
        module
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompileContext<'a> {
    pub types: &'a TypeContext,
    pub target: &'a TargetSpec,
}
