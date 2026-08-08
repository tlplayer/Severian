use crate::iree::tiling::{TileLevel, TilingPlan};
use severian_hir::{Expression, Function, Instruction};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorizationKind {
    Elementwise,
    Reduction,
    Contraction,
    Generic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorizationCandidate {
    pub instruction_index: usize,
    pub kind: VectorizationKind,
    pub preferred_width: usize,
    pub scalable: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VectorizationPlan {
    pub candidates: Vec<VectorizationCandidate>,
}

pub fn analyze_vectorization(
    function: &Function,
    tiling: &TilingPlan,
) -> VectorizationPlan {
    let default_width = tiling
        .dispatches
        .iter()
        .flat_map(|dispatch| &dispatch.levels)
        .find(|level| level.level == TileLevel::Vector)
        .and_then(|level| level.sizes.last().copied())
        .filter(|width| *width > 1)
        .unwrap_or(8);

    let mut candidates = Vec::new();

    for (instruction_index, instruction) in function.instructions.iter().enumerate() {
        let Some(expression) = value_expression(instruction) else {
            continue;
        };

        let Some(kind) = classify(expression) else {
            continue;
        };

        candidates.push(VectorizationCandidate {
            instruction_index,
            kind,
            preferred_width: width_for(kind, default_width),
            scalable: false,
        });
    }

    VectorizationPlan { candidates }
}

fn value_expression(instruction: &Instruction) -> Option<&Expression> {
    match instruction {
        Instruction::Let { value, .. }
        | Instruction::TryLet { value, .. }
        | Instruction::Evaluate(value)
        | Instruction::Assign { value, .. } => Some(value),
        _ => None,
    }
}

fn classify(expression: &Expression) -> Option<VectorizationKind> {
    match expression {
        Expression::Binary { .. } | Expression::Unary { .. } => {
            Some(VectorizationKind::Elementwise)
        }

        Expression::FusedPipeline { .. } => Some(VectorizationKind::Generic),

        Expression::Call { function, .. } => classify_name(function),

        Expression::MethodCall { method, .. } => classify_name(method),

        Expression::ListComprehension { .. }
        | Expression::SetComprehension { .. }
        | Expression::MapComprehension { .. } => Some(VectorizationKind::Generic),

        _ => None,
    }
}

fn classify_name(name: &str) -> Option<VectorizationKind> {
    let name = name.to_ascii_lowercase();

    if ["matmul", "dot", "gemm", "conv"]
        .iter()
        .any(|token| name.contains(token))
    {
        return Some(VectorizationKind::Contraction);
    }

    if ["sum", "mean", "reduce", "max", "min", "softmax"]
        .iter()
        .any(|token| name.contains(token))
    {
        return Some(VectorizationKind::Reduction);
    }

    if [
        "relu",
        "gelu",
        "add",
        "mul",
        "sub",
        "div",
        "broadcast",
        "tensor",
    ]
    .iter()
    .any(|token| name.contains(token))
    {
        return Some(VectorizationKind::Elementwise);
    }

    None
}

fn width_for(kind: VectorizationKind, default_width: usize) -> usize {
    match kind {
        VectorizationKind::Contraction => default_width.max(8),
        VectorizationKind::Reduction => default_width.max(4),
        VectorizationKind::Elementwise | VectorizationKind::Generic => default_width,
    }
}
