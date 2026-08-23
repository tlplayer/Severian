use crate::{AssertionOrigin, Block, CoveragePoint, MatchArm, ValueId};
use severian_artifact::ArtifactId;
use severian_universal::{BinaryOperator, LiteralValue, UnaryOperator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Coverage {
        point: CoveragePoint,
    },
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
    Aggregate {
        class: severian_universal::TypeId,
        fields: Vec<ValueId>,
        result: ValueId,
    },
    FieldGet {
        object: ValueId,
        field: u32,
        result: ValueId,
    },
    FieldSet {
        object: ValueId,
        field: u32,
        value: ValueId,
        result: ValueId,
    },
    Assign {
        target: ValueId,
        value: ValueId,
    },
    Call {
        function: severian_hir::FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
    },
    Spawn {
        function: severian_hir::FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
        owner: severian_hir::TaskOwner,
        locked: bool,
    },
    Await {
        task: ValueId,
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
    While {
        condition_block: Block,
        condition: ValueId,
        body: Block,
    },
    Break,
    Continue,
    Match {
        subject: ValueId,
        arms: Vec<MatchArm>,
    },
    /// Planner-produced bridge back into the standard pipeline. Source MIR
    /// never creates this operation directly.
    CompiledRegionCall {
        artifact: ArtifactId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
}
