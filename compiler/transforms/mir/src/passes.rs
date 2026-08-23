use crate::{analyze_ownership, elaborate_drops, verify, CfgBody, Module};
use severian_universal::UniversalContext;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrStage {
    Constructed,
    OwnershipChecked,
    DropElaborated,
    Optimized,
    LoweringReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    Module,
    Function,
    Region,
    Operation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnalysisId(pub u128);

#[derive(Debug, Clone)]
pub struct PassMetadata {
    pub name: &'static str,
    pub kind: PassKind,
    pub accepted_stage: IrStage,
    pub required_analyses: BTreeSet<AnalysisId>,
    pub preserved_analyses: BTreeSet<AnalysisId>,
    pub produced_stage: IrStage,
    pub parallel: bool,
    pub deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassError {
    pub pass: &'static str,
    pub message: String,
}

impl std::fmt::Display for PassError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MIR pass {} failed: {}", self.pass, self.message)
    }
}

impl std::error::Error for PassError {}

pub struct PassContext<'a> {
    pub universal: &'a UniversalContext,
}

#[derive(Default)]
pub struct AnalysisManager {
    cache: BTreeMap<AnalysisId, Box<dyn Any + Send + Sync>>,
}

impl AnalysisManager {
    pub fn get<T: Any + Send + Sync>(&self, id: AnalysisId) -> Option<&T> {
        self.cache.get(&id)?.downcast_ref()
    }

    pub fn get_or_compute<T: Any + Send + Sync>(
        &mut self,
        id: AnalysisId,
        compute: impl FnOnce() -> T,
    ) -> &T {
        self.cache.entry(id).or_insert_with(|| Box::new(compute()));
        self.get(id)
            .expect("analysis was inserted with the requested type")
    }

    pub fn invalidate_except(&mut self, preserved: &BTreeSet<AnalysisId>) {
        self.cache
            .retain(|analysis, _| preserved.contains(analysis));
    }
}

pub trait Pass: Send + Sync {
    fn metadata(&self) -> &PassMetadata;

    fn run_module(
        &self,
        _module: &mut Module,
        _context: &PassContext<'_>,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        Ok(())
    }

    fn run_function(
        &self,
        _body: &mut CfgBody,
        _context: &PassContext<'_>,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        Ok(())
    }
}

#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn add(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    pub fn run(
        &self,
        module: &mut Module,
        context: &PassContext<'_>,
        stage: &mut IrStage,
    ) -> Result<(), PassError> {
        let mut analyses = AnalysisManager::default();
        for pass in &self.passes {
            let metadata = pass.metadata();
            if metadata.accepted_stage != *stage {
                return Err(PassError {
                    pass: metadata.name,
                    message: format!(
                        "pass accepts {:?}, current stage is {:?}",
                        metadata.accepted_stage, stage
                    ),
                });
            }
            match metadata.kind {
                PassKind::Module => pass.run_module(module, context, &mut analyses)?,
                PassKind::Function | PassKind::Region | PassKind::Operation => {
                    pass.run_function(&mut module.initializer, context, &mut analyses)?;
                    for function in &mut module.functions {
                        if let Some(body) = &mut function.body {
                            pass.run_function(body, context, &mut analyses)?;
                        }
                    }
                }
            }
            analyses.invalidate_except(&metadata.preserved_analyses);
            *stage = metadata.produced_stage;
        }
        Ok(())
    }
}

struct VerifyPass {
    metadata: PassMetadata,
}

impl VerifyPass {
    fn new(accepted_stage: IrStage, produced_stage: IrStage) -> Self {
        Self {
            metadata: PassMetadata {
                name: "verify",
                kind: PassKind::Module,
                accepted_stage,
                required_analyses: BTreeSet::new(),
                preserved_analyses: BTreeSet::new(),
                produced_stage,
                parallel: false,
                deterministic: true,
            },
        }
    }
}

impl Pass for VerifyPass {
    fn metadata(&self) -> &PassMetadata {
        &self.metadata
    }

    fn run_module(
        &self,
        module: &mut Module,
        context: &PassContext<'_>,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        verify(module, context.universal).map_err(|error| PassError {
            pass: self.metadata.name,
            message: error.to_string(),
        })
    }
}

struct DropElaborationPass {
    metadata: PassMetadata,
}

impl DropElaborationPass {
    fn new() -> Self {
        Self {
            metadata: PassMetadata {
                name: "drop-elaboration",
                kind: PassKind::Function,
                accepted_stage: IrStage::Constructed,
                required_analyses: BTreeSet::new(),
                preserved_analyses: BTreeSet::new(),
                produced_stage: IrStage::DropElaborated,
                parallel: false,
                deterministic: true,
            },
        }
    }
}

impl Pass for DropElaborationPass {
    fn metadata(&self) -> &PassMetadata {
        &self.metadata
    }

    fn run_function(
        &self,
        body: &mut CfgBody,
        context: &PassContext<'_>,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        elaborate_drops(body, &context.universal.types).map_err(|errors| PassError {
            pass: self.metadata.name,
            message: errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        })
    }
}

struct OwnershipPass {
    metadata: PassMetadata,
}

impl OwnershipPass {
    fn new() -> Self {
        Self {
            metadata: PassMetadata {
                name: "ownership",
                kind: PassKind::Function,
                accepted_stage: IrStage::DropElaborated,
                required_analyses: BTreeSet::new(),
                preserved_analyses: BTreeSet::new(),
                produced_stage: IrStage::OwnershipChecked,
                parallel: false,
                deterministic: true,
            },
        }
    }
}

impl Pass for OwnershipPass {
    fn metadata(&self) -> &PassMetadata {
        &self.metadata
    }

    fn run_function(
        &self,
        body: &mut CfgBody,
        context: &PassContext<'_>,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        analyze_ownership(body, &context.universal.types)
            .map(|_| ())
            .map_err(|errors| PassError {
                pass: self.metadata.name,
                message: errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

pub fn run_required_pipeline(
    module: &mut Module,
    universal: &UniversalContext,
) -> Result<IrStage, PassError> {
    let context = PassContext { universal };
    let mut stage = IrStage::Constructed;
    let mut manager = PassManager::default();
    manager.add(VerifyPass::new(IrStage::Constructed, IrStage::Constructed));
    manager.add(DropElaborationPass::new());
    manager.add(OwnershipPass::new());
    manager.add(VerifyPass::new(
        IrStage::OwnershipChecked,
        IrStage::LoweringReady,
    ));
    manager.run(module, &context, &mut stage)?;
    Ok(stage)
}
