use crate::{CompileContext, CompileError, CompilePlan, CompileRegion, PlanSegment};
use severian_artifact::ArtifactId;
use severian_mlir::{verify_artifact, MlirArtifact, VerifiedMlirArtifact};
use severian_universal::CompilerId;
use std::collections::HashMap;

pub trait CompileHandler: Send + Sync {
    fn compile(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<MlirArtifact, CompileError>;
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
    ) -> Result<Vec<VerifiedMlirArtifact>, CompileError> {
        plan.segments
            .iter()
            .filter_map(|segment| match segment {
                PlanSegment::Compiler(region) => Some(region),
                PlanSegment::Standard(_) => None,
            })
            .map(|region| {
                let handler = self
                    .handlers
                    .get(&region.compiler)
                    .ok_or(CompileError::MissingHandler(region.compiler))?;
                let artifact = handler.compile(region, context)?;
                if artifact.inputs.len() != region.inputs.len()
                    || artifact.outputs.len() != region.outputs.len()
                {
                    return Err(CompileError::InvalidArtifact(format!(
                        "handler output does not match region {:?}",
                        region.id
                    )));
                }
                verify_artifact(ArtifactId::for_region(region.id), artifact, context.target)
                    .map_err(|error| CompileError::InvalidArtifact(error.to_string()))
            })
            .collect()
    }
}
