use crate::ValueId;
use severian_artifact::ArtifactId;
use severian_universal::{BinaryOperator, LiteralValue, UnaryOperator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Constant {
        value: LiteralValue,
        result: ValueId,
    },
    Unary {
        operator: UnaryOperator,
        operand: ValueId,
        result: ValueId,
    },
    Binary {
        operator: BinaryOperator,
        left: ValueId,
        right: ValueId,
        result: ValueId,
    },
    /// Planner-produced bridge back into the standard pipeline. Source MIR
    /// never creates this operation directly.
    CompiledRegionCall {
        artifact: ArtifactId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
}
