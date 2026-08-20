use crate::ValueId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    ConstantInt {
        value: i64,
        result: ValueId,
    },
    AddInt {
        left: ValueId,
        right: ValueId,
        result: ValueId,
    },
}
