use crate::{
    CompileContext, CompileError, CompilePlan, CompileRegion, CompiledRegionArtifact, PlanSegment,
    VerifiedCompiledRegionArtifact, VerifiedGpuKernelBundle,
};
use severian_artifact::ArtifactId;
use severian_mlir::verify_artifact;
use severian_universal::CompilerId;
use std::collections::HashMap;

pub trait CompileHandler: Send + Sync {
    fn compile(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<CompiledRegionArtifact, CompileError>;
}

#[derive(Default)]
pub struct CompilerRegistry {
    handlers: HashMap<CompilerId, Box<dyn CompileHandler>>,
}

impl CompilerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        compiler: CompilerId,
        handler: impl CompileHandler + 'static,
    ) -> Result<(), CompileError> {
        if self.handlers.contains_key(&compiler) {
            return Err(CompileError::DuplicateHandler(compiler));
        }
        self.handlers.insert(compiler, Box::new(handler));
        Ok(())
    }

    pub fn compile(
        &self,
        plan: &CompilePlan,
        context: &CompileContext<'_>,
    ) -> Result<Vec<VerifiedCompiledRegionArtifact>, CompileError> {
        plan.initializer
            .segments
            .iter()
            .chain(
                plan.functions
                    .iter()
                    .filter_map(|function| function.body.as_ref())
                    .flat_map(|body| &body.segments),
            )
            .filter_map(|segment| match segment {
                PlanSegment::Compiler(region) => Some(region),
                PlanSegment::Standard(_) => None,
            })
            .chain(&plan.nested_regions)
            .map(|region| {
                let handler = self
                    .handlers
                    .get(&region.compiler)
                    .ok_or(CompileError::MissingHandler(region.compiler))?;
                let id = ArtifactId::for_region(region.id);
                match handler.compile(region, context)? {
                    CompiledRegionArtifact::CpuMlir(artifact) => {
                        validate_arity(region, artifact.inputs.len(), artifact.outputs.len())?;
                        verify_artifact(id, artifact, context.target)
                            .map(VerifiedCompiledRegionArtifact::CpuMlir)
                            .map_err(|error| CompileError::InvalidArtifact(error.to_string()))
                    }
                    CompiledRegionArtifact::GpuKernel(bundle) => {
                        validate_arity(region, bundle.inputs.len(), bundle.outputs.len())?;
                        if bundle.architecture.is_empty() {
                            return Err(CompileError::InvalidArtifact(
                                "GPU kernel bundle has no target architecture".into(),
                            ));
                        }
                        if bundle.plan.node_regions.len() != bundle.graph.nodes().len() {
                            return Err(CompileError::InvalidArtifact(
                                "GPU fusion plan does not map the complete graph".into(),
                            ));
                        }
                        Ok(VerifiedCompiledRegionArtifact::GpuKernel(
                            VerifiedGpuKernelBundle { id, bundle },
                        ))
                    }
                }
            })
            .collect()
    }
}

fn validate_arity(
    region: &CompileRegion,
    inputs: usize,
    outputs: usize,
) -> Result<(), CompileError> {
    if inputs != region.inputs.len() || outputs != region.outputs.len() {
        return Err(CompileError::InvalidArtifact(format!(
            "handler output does not match region {:?}",
            region.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EffectSet, GpuKernelBundle, GpuTarget, PlannedBlock};
    use severian_artifact::CompiledRegionId;
    use severian_fusion::{plan, DeviceModel, FusionGraph, FusionNode, NodeId, NodeKind, Shape};
    use severian_mir::Module;
    use severian_target::{Device, DeviceKind, FeatureSet, TargetSpec};
    use severian_universal::{CompilerId, ExecutionPlacement, TypeContextBuilder};

    struct GpuOnlyHandler;

    impl CompileHandler for GpuOnlyHandler {
        fn compile(
            &self,
            _: &CompileRegion,
            _: &CompileContext<'_>,
        ) -> Result<CompiledRegionArtifact, CompileError> {
            let graph = FusionGraph::new(vec![
                FusionNode::structural(0, NodeKind::Parameter, [], Shape::ranked([8], 32)),
                FusionNode::structural(
                    1,
                    NodeKind::Elementwise,
                    [NodeId(0)],
                    Shape::ranked([8], 32),
                ),
            ])
            .unwrap();
            let plan = plan(&graph, DeviceModel::conservative_gpu());
            Ok(CompiledRegionArtifact::GpuKernel(GpuKernelBundle {
                target: GpuTarget::Amd,
                architecture: "gfx1100".into(),
                graph,
                plan,
                inputs: Vec::new(),
                outputs: Vec::new(),
            }))
        }
    }

    #[test]
    fn gpu_artifacts_bypass_mlir_artifact_verification() {
        let compiler = CompilerId::from_path("test.gpu");
        let region = CompileRegion {
            id: CompiledRegionId::new(0),
            compiler,
            operations: Vec::new(),
            compile_operations: Vec::new(),
            output_slots: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
            placement: Some(ExecutionPlacement::Gpu),
        };
        let plan = CompilePlan {
            source: Module::default(),
            initializer: PlannedBlock {
                segments: vec![PlanSegment::Compiler(region)],
            },
            functions: Vec::new(),
            nested_regions: Vec::new(),
        };
        let types = TypeContextBuilder::new().build();
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: FeatureSet::from_names(["vendor.amd"]),
        });
        let mut registry = CompilerRegistry::new();
        registry.register(compiler, GpuOnlyHandler).unwrap();

        let artifacts = registry
            .compile(
                &plan,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap();
        assert!(matches!(
            artifacts.as_slice(),
            [VerifiedCompiledRegionArtifact::GpuKernel(_)]
        ));
    }
}
