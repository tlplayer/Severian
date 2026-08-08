use severian_hir::{Expression, Function, Instruction, MatchPattern};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopFusionCandidate {
    pub first_index: usize,
    pub second_index: usize,
}

pub fn find_loop_fusion_candidates(function: &Function) -> Vec<LoopFusionCandidate> {
    let mut result = Vec::new();

    for index in 0..function.instructions.len().saturating_sub(1) {
        let first = &function.instructions[index];
        let second = &function.instructions[index + 1];

        if compatible_loops(first, second) {
            result.push(LoopFusionCandidate {
                first_index: index,
                second_index: index + 1,
            });
        }
    }

    result
}

fn compatible_loops(first: &Instruction, second: &Instruction) -> bool {
    match (first, second) {
        (
            Instruction::For {
                setup: None,
                pattern: first_pattern,
                iterable: first_iterable,
                ..
            },
            Instruction::For {
                setup: None,
                pattern: second_pattern,
                iterable: second_iterable,
                ..
            },
        ) => {
            structurally_equal_pattern(first_pattern, second_pattern)
                && side_effect_free_iterable(first_iterable)
                && first_iterable == second_iterable
        }

        _ => false,
    }
}

fn structurally_equal_pattern(left: &MatchPattern, right: &MatchPattern) -> bool {
    left == right
}

fn side_effect_free_iterable(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Variable(_)
            | Expression::List(_)
            | Expression::Tuple(_)
            | Expression::Set(_)
            | Expression::Map(_)
    )
}
