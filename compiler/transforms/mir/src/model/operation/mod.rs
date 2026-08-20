use crate::ValueId;
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
}
