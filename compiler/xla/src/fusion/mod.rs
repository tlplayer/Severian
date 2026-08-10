mod instruction_fusion;
mod loop_fusion;
mod multi_output_fusion;

pub use instruction_fusion::{
    find_instruction_fusion_candidates, FusionCandidate, FusionKind, InstructionFusion,
};
pub use loop_fusion::{find_loop_fusion_candidates, LoopFusionCandidate};
pub use multi_output_fusion::{find_multi_output_candidates, MultiOutputFusionCandidate};

use severian_hir::Function;

#[derive(Debug, Clone, Default)]
pub struct FusionPlan {
    pub instructions: Vec<FusionCandidate>,
    pub loops: Vec<LoopFusionCandidate>,
    pub multi_output: Vec<MultiOutputFusionCandidate>,
}

pub fn analyze_function(function: &Function) -> FusionPlan {
    FusionPlan {
        instructions: find_instruction_fusion_candidates(function),
        loops: find_loop_fusion_candidates(function),
        multi_output: find_multi_output_candidates(function),
    }
}
