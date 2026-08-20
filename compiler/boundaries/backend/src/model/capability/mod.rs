#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweredType {
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    ConstantI64 {
        value: i64,
        result: ValueId,
    },
    AddI64 {
        left: ValueId,
        right: ValueId,
        result: ValueId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoweredModule {
    pub values: Vec<LoweredType>,
    pub operations: Vec<Operation>,
    pub last_binding: Option<ValueId>,
}
