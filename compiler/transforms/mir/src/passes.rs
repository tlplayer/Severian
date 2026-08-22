use crate::{CfgBody, Module};
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
        self.get(id).expect("analysis was inserted with the requested type")
    }

    pub fn invalidate_except(&mut self, preserved: &BTreeSet<AnalysisId>) {
        self.cache.retain(|analysis, _| preserved.contains(analysis));
    }
}

pub trait Pass: Send + Sync {
    fn metadata(&self) -> &PassMetadata;

    fn run_module(
        &self,
        _module: &mut Module,
        _analyses: &mut AnalysisManager,
    ) -> Result<(), PassError> {
        Ok(())
    }

    fn run_function(
        &self,
        _body: &mut CfgBody,
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

    pub fn run(&self, module: &mut Module, stage: &mut IrStage) -> Result<(), PassError> {
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
                PassKind::Module => pass.run_module(module, &mut analyses)?,
                PassKind::Function | PassKind::Region | PassKind::Operation => {
                    pass.run_function(&mut module.initializer_cfg, &mut analyses)?;
                    for function in &mut module.functions {
                        if let Some(body) = &mut function.cfg {
                            pass.run_function(body, &mut analyses)?;
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
