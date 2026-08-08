mod tile;

pub use tile::{
    choose_tile_sizes, TileLevel, TilePlan, TiledDispatch, TilingPlan, TilingTarget,
};

use super::dispatch::DispatchPlan;
use severian_hir::Function;

pub fn analyze_function(function: &Function, dispatch: &DispatchPlan) -> TilingPlan {
    tile::plan_function_tiling(function, dispatch)
}
