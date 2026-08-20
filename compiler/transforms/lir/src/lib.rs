#![forbid(unsafe_code)]

use severian_universal::{BinaryOperator, LiteralValue, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweredFloatFormat {
    Ieee(u16),
    BrainFloat16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweredType {
    Integer { bits: u16, signed: bool },
    Float { format: LoweredFloatFormat },
    Boolean,
    String,
    Bytes,
    None,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    pub id: ValueId,
    pub ty: LoweredType,
}

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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub operations: Vec<Operation>,
    pub last_binding: Option<ValueId>,
}
