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
    Aggregate(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaceBase {
    Local(LocalId),
    Global(GlobalId),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place {
    pub base: PlaceBase,
    pub projection: Vec<Projection>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Projection {
    Field(u32),
    Index(LocalId),
    Dereference,
    Downcast(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalDecl {
    pub id: GlobalId,
    pub ty: LoweredType,
    pub mutable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub ty: LoweredType,
    pub mutable: bool,
    pub argument: bool,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwner {
    SelfScope,
    Runtime,
    Inferred,
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
    Aggregate {
        class: u32,
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
    Load {
        place: Place,
        result: ValueId,
    },
    Store {
        place: Place,
        value: ValueId,
    },
    Call {
        function: FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
    },
    Spawn {
        function: FunctionId,
        arguments: Vec<ValueId>,
        result: ValueId,
        owner: TaskOwner,
        locked: bool,
    },
    Await {
        task: ValueId,
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
    While {
        condition_block: Block,
        condition: ValueId,
        body: Block,
    },
    Break,
    Continue,
    ArtifactCall {
        artifact: ArtifactId,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Goto(BlockId),
    Branch {
        condition: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    },
    Switch {
        discriminant: ValueId,
        targets: Vec<(Case, BlockId)>,
        fallback: BlockId,
    },
    Call {
        function: FunctionId,
        arguments: Vec<ValueId>,
        destination: Option<Place>,
        target: BlockId,
    },
    Return(Option<ValueId>),
    Throw(ValueId),
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Case {
    Integer(i128),
    Boolean(bool),
    Variant(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub operations: Vec<Operation>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgBody {
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
    pub locals: Vec<LocalDecl>,
    pub return_type: LoweredType,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    /// Structured fields remain readable while downstream emitters migrate;
    /// CFG lowering populates `storage_globals` and `initializer_cfg`.
    pub values: Vec<Value>,
    pub globals: Vec<ValueId>,
    pub initializer: Block,
    pub functions: Vec<Function>,
    pub entry: Option<FunctionId>,
    pub traits: Vec<TraitDeclaration>,
    pub classes: Vec<ClassDeclaration>,
    pub storage_globals: Vec<GlobalDecl>,
    pub initializer_cfg: Option<CfgBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassFieldDeclaration {
    pub name: String,
    pub ty: LoweredType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDeclaration {
    pub id: u32,
    pub name: String,
    pub fields: Vec<ClassFieldDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitMethodDeclaration {
    pub name: String,
    pub parameters: Vec<TraitType>,
    pub result: TraitType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitType {
    SelfType,
    Concrete(LoweredType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitDeclaration {
    pub id: TraitId,
    pub name: String,
    pub methods: Vec<TraitMethodDeclaration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraitId {
    pub package: u128,
    pub module: u128,
    pub declaration: u128,
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
    /// Physical signature and executable CFG produced by the authoritative
    /// MIR CFG lowering path.
    pub parameter_types: Vec<LoweredType>,
    pub cfg: Option<CfgBody>,
}
