use crate::iree::dispatch::{DispatchId, DispatchPlan, DispatchRoot};
use severian_hir::Function;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingTarget {
    Cpu,
    Gpu,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLevel {
    Workgroup,
    Parallel,
    Reduction,
    Vector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub level: TileLevel,
    pub sizes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiledDispatch {
    pub dispatch: DispatchId,
    pub target: TilingTarget,
    pub levels: Vec<TilePlan>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TilingPlan {
    pub dispatches: Vec<TiledDispatch>,
}

pub fn plan_function_tiling(
    _function: &Function,
    dispatch_plan: &DispatchPlan,
) -> TilingPlan {
    let dispatches = dispatch_plan
        .active_regions()
        .map(|region| {
            let target = infer_target(region.root);
            let levels = choose_tile_sizes(region.root, target);

            TiledDispatch {
                dispatch: region.id,
                target,
                levels,
            }
        })
        .collect();

    TilingPlan { dispatches }
}

pub fn choose_tile_sizes(root: DispatchRoot, target: TilingTarget) -> Vec<TilePlan> {
    match (root, target) {
        (DispatchRoot::TensorCall, TilingTarget::Gpu)
        | (DispatchRoot::ReductionLike, TilingTarget::Gpu) => vec![
            TilePlan {
                level: TileLevel::Workgroup,
                sizes: vec![128, 128, 32],
            },
            TilePlan {
                level: TileLevel::Parallel,
                sizes: vec![32, 32, 8],
            },
            TilePlan {
                level: TileLevel::Reduction,
                sizes: vec![1, 1, 8],
            },
        ],

        (DispatchRoot::TensorCall, TilingTarget::Cpu)
        | (DispatchRoot::ReductionLike, TilingTarget::Cpu) => vec![
            TilePlan {
                level: TileLevel::Parallel,
                sizes: vec![64, 64, 32],
            },
            TilePlan {
                level: TileLevel::Reduction,
                sizes: vec![1, 1, 8],
            },
            TilePlan {
                level: TileLevel::Vector,
                sizes: vec![8, 8, 1],
            },
        ],

        (DispatchRoot::Elementwise, TilingTarget::Gpu)
        | (DispatchRoot::FusedPipeline, TilingTarget::Gpu) => vec![TilePlan {
            level: TileLevel::Workgroup,
            sizes: vec![256],
        }],

        (DispatchRoot::Elementwise, _)
        | (DispatchRoot::FusedPipeline, _)
        | (DispatchRoot::Comprehension, _) => vec![TilePlan {
            level: TileLevel::Vector,
            sizes: vec![8],
        }],

        _ => vec![TilePlan {
            level: TileLevel::Parallel,
            sizes: vec![32],
        }],
    }
}

fn infer_target(root: DispatchRoot) -> TilingTarget {
    match root {
        DispatchRoot::TensorCall | DispatchRoot::ReductionLike => TilingTarget::Generic,
        DispatchRoot::Elementwise
        | DispatchRoot::FusedPipeline
        | DispatchRoot::Comprehension => TilingTarget::Generic,
    }
}
