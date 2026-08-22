#![forbid(unsafe_code)]

use severian_artifact::ArtifactId;

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
    Arguments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    pub id: ValueId,
    pub ty: LoweredType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub expression: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constant {
    Integer(String),
    Float(String),
    Boolean(bool),
    String(String),
    Bytes(Vec<u8>),
    None,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperation {
    Positive,
    Negative,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Contains,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Coverage {
        key: String,
    },
    Constant {
        value: Constant,
        result: ValueId,
    },
    Unary {
        operator: UnaryOperation,
        operand: ValueId,
        result: ValueId,
    },
    Binary {
        operator: BinaryOperation,
        left: ValueId,
        right: ValueId,
        result: ValueId,
    },
    Call {
        function: FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
    },
    /// A call to the versioned native runtime ABI selected during lowering.
    /// Emitters treat the symbol and physical signature generically.
    RuntimeCall {
        symbol: String,
        arguments: Vec<ValueId>,
        result: Option<ValueId>,
    },
    Return {
        value: Option<ValueId>,
    },
    Assert {
        condition: ValueId,
        message: Option<ValueId>,
        location: Option<AssertionLocation>,
    },
    If {
        condition: ValueId,
        then_block: Block,
        else_block: Block,
    },
    ArtifactCall {
        artifact: ArtifactId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub globals: Vec<ValueId>,
    pub initializer: Block,
    pub functions: Vec<Function>,
    pub entry: Option<FunctionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionLinkage {
    Internal,
    External { symbol: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub parameters: Vec<ValueId>,
    pub result: LoweredType,
    pub body: Option<Block>,
    pub linkage: FunctionLinkage,
}
