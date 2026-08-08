mod vectorize;

pub use vectorize::{
    analyze_vectorization, VectorizationCandidate, VectorizationKind, VectorizationPlan,
};

use super::tiling::TilingPlan;
use severian_hir::Function;

pub fn analyze_function(
    function: &Function,
    tiling: &TilingPlan,
) -> VectorizationPlan {
    analyze_vectorization(function, tiling)
}
