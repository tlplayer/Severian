use crate::{AssertionOrigin, Block, ValueId};
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
    Call {
        function: severian_hir::FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
    },
    Return {
        value: Option<ValueId>,
    },
    Assert {
        condition: ValueId,
        message: Option<ValueId>,
        origin: AssertionOrigin,
    },
    If {
        condition: ValueId,
        then_block: Block,
        else_block: Block,
    },
    /// Planner-produced bridge back into the standard pipeline. Source MIR
    /// never creates this operation directly.
    CompiledRegionCall {
        artifact: ArtifactId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
}
