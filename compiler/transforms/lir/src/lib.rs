#![forbid(unsafe_code)]

use severian_artifact::ArtifactId;
use severian_source::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweredFloatFormat {
    Float8E4M3Fn,
    Float8E5M2,
    Ieee(u16),
    BrainFloat16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweredTensorElement {
    Integer { bits: u16, signed: bool },
    Float { format: LoweredFloatFormat },
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweredTensorDimension {
    Dynamic,
    Known(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweredTensorShape {
    Unranked,
    Ranked(Vec<LoweredTensorDimension>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Tensor {
        element: LoweredTensorElement,
        shape: LoweredTensorShape,
    },
    Task(TaskValueType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskValueType {
    Integer { bits: u16, signed: bool },
    Float { format: LoweredFloatFormat },
    Boolean,
    String,
    Bytes,
    None,
    Unit,
    Arguments,
    Aggregate(u32),
    Tensor {
        element: LoweredTensorElement,
        shape: LoweredTensorShape,
    },
}

impl LoweredType {
    pub fn task(self) -> Option<Self> {
        Some(Self::Task(match self {
            Self::Integer { bits, signed } => TaskValueType::Integer { bits, signed },
            Self::Float { format } => TaskValueType::Float { format },
            Self::Boolean => TaskValueType::Boolean,
            Self::String => TaskValueType::String,
            Self::Bytes => TaskValueType::Bytes,
            Self::None => TaskValueType::None,
            Self::Unit => TaskValueType::Unit,
            Self::Arguments => TaskValueType::Arguments,
            Self::Aggregate(id) => TaskValueType::Aggregate(id),
            Self::Tensor { element, shape } => TaskValueType::Tensor { element, shape },
            Self::Task(_) => return None,
        }))
    }

    pub fn task_result(self) -> Option<Self> {
        let Self::Task(result) = self else {
            return None;
        };
        Some(match result {
            TaskValueType::Integer { bits, signed } => Self::Integer { bits, signed },
            TaskValueType::Float { format } => Self::Float { format },
            TaskValueType::Boolean => Self::Boolean,
            TaskValueType::String => Self::String,
            TaskValueType::Bytes => Self::Bytes,
            TaskValueType::None => Self::None,
            TaskValueType::Unit => Self::Unit,
            TaskValueType::Arguments => Self::Arguments,
            TaskValueType::Aggregate(id) => Self::Aggregate(id),
            TaskValueType::Tensor { element, shape } => Self::Tensor { element, shape },
        })
    }
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
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub ty: LoweredType,
    pub mutable: bool,
    pub argument: bool,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    BitwiseOr,
    BitwiseAnd,
    BitwiseXor,
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
    Convert {
        operand: ValueId,
        result: ValueId,
        kind: severian_universal::ConversionKind,
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
    AddressOf {
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
    SpawnFieldUpdate {
        place: Place,
        operator: BinaryOperation,
        value: ValueId,
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
    /// Execution domain retained from universal/HIR placement syntax.
    pub execution: Option<severian_universal::ExecutionPlacement>,
    pub operations: Vec<Operation>,
    pub operation_spans: Vec<Option<Span>>,
    pub terminator: Terminator,
    pub terminator_span: Option<Span>,
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
    /// Concrete accelerator architecture selected by the compiler component
    /// resolver (for example `gfx1100`).
    pub gpu_architecture: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraitType {
    SelfType,
    Concrete(LoweredType),
    Symbolic(String),
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
