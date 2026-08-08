mod formation;
mod fusion;

pub use formation::{
    analyze_dispatches, DispatchId, DispatchPlan, DispatchRegion, DispatchRoot,
};
pub use fusion::{fuse_dispatches, DispatchFusion};

use severian_hir::Function;

pub fn analyze_function(function: &Function) -> DispatchPlan {
    let mut plan = analyze_dispatches(function);
    let fusions = fuse_dispatches(function, &plan);

    for fusion in &fusions {
        plan.mark_fused(fusion);
    }

    plan.fusions = fusions;
    plan
}
