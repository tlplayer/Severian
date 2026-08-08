mod bufferize;

pub use bufferize::{
    analyze_bufferization, AllocationKind, BufferizationPlan, BufferizedValue,
    BufferizationStrategy,
};

use super::dispatch::DispatchPlan;
use severian_hir::Function;

pub fn analyze_function(
    function: &Function,
    dispatch: &DispatchPlan,
) -> BufferizationPlan {
    analyze_bufferization(function, dispatch)
}
