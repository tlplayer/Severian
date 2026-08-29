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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Invariant {
    WellFormed,
    DropsElaborated,
    OwnershipValid,
    LoweringReady,
}

pub type InvariantSet = BTreeSet<Invariant>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntityKind {
    Function,
    Global,
    BasicBlock,
    Local,
    Statement,
}

pub type EntitySet = BTreeSet<EntityKind>;

#[derive(Debug, Clone, Default)]
pub struct PassContract {
    pub requires: InvariantSet,
    pub preserves: InvariantSet,
    pub establishes: InvariantSet,
    pub may_remove: EntitySet,
    pub may_introduce: EntitySet,
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
    pub contract: PassContract,
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
        let mut invariants = stage_invariants(*stage);
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
            let missing = metadata
                .contract
                .requires
                .difference(&invariants)
                .copied()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(PassError {
                    pass: metadata.name,
                    message: format!("required invariants are not established: {missing:?}"),
                });
            }
            let unavailable_preservation = metadata
                .contract
                .preserves
                .difference(&invariants)
                .copied()
                .collect::<Vec<_>>();
            if !unavailable_preservation.is_empty() {
                return Err(PassError {
                    pass: metadata.name,
                    message: format!(
                        "pass claims to preserve unavailable invariants: {unavailable_preservation:?}"
                    ),
                });
            }
            let before = entity_counts(module);
            match metadata.kind {
                PassKind::Module => pass.run_module(module, context, &mut analyses)?,
                PassKind::Function | PassKind::Region | PassKind::Operation => {
                    pass.run_function(&mut module.initializer, context, &mut analyses)
                        .map_err(|mut error| {
                            error.message = format!("in module initializer: {}", error.message);
                            error
                        })?;
                    for function in &mut module.functions {
                        if let Some(body) = &mut function.body {
                            pass.run_function(body, context, &mut analyses).map_err(
                                |mut error| {
                                    error.message = format!(
                                        "in function `{}`: {}",
                                        function.name, error.message
                                    );
                                    error
                                },
                            )?;
                        }
                    }
                }
            }
            enforce_entity_contract(metadata, &before, &entity_counts(module))?;
            let must_verify = metadata.contract.preserves.contains(&Invariant::WellFormed)
                || metadata
                    .contract
                    .establishes
                    .contains(&Invariant::WellFormed);
            if must_verify {
                verify(module, context.universal).map_err(|error| PassError {
                    pass: metadata.name,
                    message: format!("post-pass MIR verification failed: {error}"),
                })?;
            }
            analyses.invalidate_except(&metadata.preserved_analyses);
            invariants.retain(|invariant| metadata.contract.preserves.contains(invariant));
            invariants.extend(metadata.contract.establishes.iter().copied());
            *stage = metadata.produced_stage;
        }
        Ok(())
    }
}

fn stage_invariants(stage: IrStage) -> InvariantSet {
    match stage {
        IrStage::Constructed => InvariantSet::new(),
        IrStage::DropElaborated => {
            BTreeSet::from([Invariant::WellFormed, Invariant::DropsElaborated])
        }
        IrStage::OwnershipChecked | IrStage::Optimized => BTreeSet::from([
            Invariant::WellFormed,
            Invariant::DropsElaborated,
            Invariant::OwnershipValid,
        ]),
        IrStage::LoweringReady => BTreeSet::from([
            Invariant::WellFormed,
            Invariant::DropsElaborated,
            Invariant::OwnershipValid,
            Invariant::LoweringReady,
        ]),
    }
}

fn entity_counts(module: &Module) -> BTreeMap<EntityKind, usize> {
    let bodies = std::iter::once(&module.initializer).chain(
        module
            .functions
            .iter()
            .filter_map(|function| function.body.as_ref()),
    );
    let mut counts = BTreeMap::from([
        (EntityKind::Function, module.functions.len()),
        (EntityKind::Global, module.globals.len()),
        (EntityKind::BasicBlock, 0),
        (EntityKind::Local, 0),
        (EntityKind::Statement, 0),
    ]);
    for body in bodies {
        *counts.entry(EntityKind::BasicBlock).or_default() += body.blocks.len();
        *counts.entry(EntityKind::Local).or_default() += body.locals.len();
        *counts.entry(EntityKind::Statement).or_default() += body
            .blocks
            .iter()
            .map(|block| block.statements.len())
            .sum::<usize>();
    }
    counts
}

fn enforce_entity_contract(
    metadata: &PassMetadata,
    before: &BTreeMap<EntityKind, usize>,
    after: &BTreeMap<EntityKind, usize>,
) -> Result<(), PassError> {
    for kind in [
        EntityKind::Function,
        EntityKind::Global,
        EntityKind::BasicBlock,
        EntityKind::Local,
        EntityKind::Statement,
    ] {
        let old = before.get(&kind).copied().unwrap_or(0);
        let new = after.get(&kind).copied().unwrap_or(0);
        if new > old && !metadata.contract.may_introduce.contains(&kind) {
            return Err(PassError {
                pass: metadata.name,
                message: format!(
                    "introduced {} {kind:?} entities without declaring it",
                    new - old
                ),
            });
        }
        if old > new && !metadata.contract.may_remove.contains(&kind) {
            return Err(PassError {
                pass: metadata.name,
                message: format!(
                    "removed {} {kind:?} entities without declaring it",
                    old - new
                ),
            });
        }
    }
    Ok(())
}

struct VerifyPass {
    metadata: PassMetadata,
}

impl VerifyPass {
    fn new(accepted_stage: IrStage, produced_stage: IrStage) -> Self {
        let contract = if accepted_stage == IrStage::Constructed {
            PassContract {
                establishes: BTreeSet::from([Invariant::WellFormed]),
                ..PassContract::default()
            }
        } else {
            PassContract {
                requires: BTreeSet::from([
                    Invariant::WellFormed,
                    Invariant::DropsElaborated,
                    Invariant::OwnershipValid,
                ]),
                preserves: BTreeSet::from([
                    Invariant::WellFormed,
                    Invariant::DropsElaborated,
                    Invariant::OwnershipValid,
                ]),
                establishes: BTreeSet::from([Invariant::LoweringReady]),
                ..PassContract::default()
            }
        };
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
                contract,
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
                contract: PassContract {
                    requires: BTreeSet::from([Invariant::WellFormed]),
                    preserves: BTreeSet::from([Invariant::WellFormed]),
                    establishes: BTreeSet::from([Invariant::DropsElaborated]),
                    may_introduce: BTreeSet::from([EntityKind::Statement]),
                    ..PassContract::default()
                },
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
                contract: PassContract {
                    requires: BTreeSet::from([Invariant::WellFormed, Invariant::DropsElaborated]),
                    preserves: BTreeSet::from([Invariant::WellFormed, Invariant::DropsElaborated]),
                    establishes: BTreeSet::from([Invariant::OwnershipValid]),
                    ..PassContract::default()
                },
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

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(contract: PassContract) -> PassMetadata {
        PassMetadata {
            name: "test",
            kind: PassKind::Module,
            accepted_stage: IrStage::Constructed,
            required_analyses: BTreeSet::new(),
            preserved_analyses: BTreeSet::new(),
            produced_stage: IrStage::Constructed,
            parallel: false,
            deterministic: true,
            contract,
        }
    }

    #[test]
    fn undeclared_entity_introduction_is_rejected() {
        let pass = metadata(PassContract::default());
        let before = BTreeMap::from([(EntityKind::Statement, 1)]);
        let after = BTreeMap::from([(EntityKind::Statement, 2)]);
        assert!(enforce_entity_contract(&pass, &before, &after).is_err());
    }

    #[test]
    fn declared_entity_introduction_is_accepted() {
        let pass = metadata(PassContract {
            may_introduce: BTreeSet::from([EntityKind::Statement]),
            ..PassContract::default()
        });
        let before = BTreeMap::from([(EntityKind::Statement, 1)]);
        let after = BTreeMap::from([(EntityKind::Statement, 2)]);
        assert!(enforce_entity_contract(&pass, &before, &after).is_ok());
    }
}
