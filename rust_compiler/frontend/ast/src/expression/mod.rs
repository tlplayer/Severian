use crate::TypeAnnotation;
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Integer(String),
    Float(String),
    Measured { magnitude: String, suffix: String },
    Boolean(bool),
    Character(char),
    String(String),
    Bytes(Vec<u8>),
    None,
    Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Positive,
    Negative,
    Not,
    Borrow,
    BorrowMut,
    AddressOf,
    Copy,
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Pipe,
    BitwiseAnd,
    BitwiseXor,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Equal,
    Identity,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Contains,
    And,
    Or,
}

/// The lifetime owner selected for a spawned task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwner {
    SelfScope,
    Runtime,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallArgument {
    pub name: Option<String>,
    pub spread: bool,
    pub value: Expression,
    /// The error shape expected by `throws(value -> ErrorType)`.
    pub expected_error: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapEntry {
    pub key: Expression,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComprehensionClause {
    pub bindings: Vec<String>,
    pub iterable: Expression,
    pub condition: Option<Expression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockCase {
    pub call: Expression,
    pub result: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Literal(Literal),
    List(Vec<Expression>),
    Set(Vec<Expression>),
    Map(Vec<MapEntry>),
    ListComprehension {
        value: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    SetComprehension {
        value: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    MapComprehension {
        key: Box<Expression>,
        value: Box<Expression>,
        clauses: Vec<ComprehensionClause>,
    },
    Mock {
        cases: Vec<MockCase>,
        fallback: Box<Expression>,
    },
    Lambda {
        parameters: Vec<String>,
        body: Box<Expression>,
    },
    Tuple(Vec<Expression>),
    Name(String),
    /// An internable compiler symbol written as `:name` (`Y` in the
    /// universal compiler-term taxonomy).
    Symbol(String),
    Member {
        object: Box<Expression>,
        name: String,
    },
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    Slice {
        object: Box<Expression>,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
        step: Option<Box<Expression>>,
        start_exclusive: bool,
        end_inclusive: bool,
    },
    TypeApplication {
        callee: Box<Expression>,
        arguments: Vec<TypeAnnotation>,
    },
    Call {
        callee: Box<Expression>,
        arguments: Vec<CallArgument>,
    },
    Async {
        expression: Box<Expression>,
        owner: TaskOwner,
        locked: bool,
    },
    Await {
        expression: Box<Expression>,
    },
    Conditional {
        value: Box<Expression>,
        condition: Box<Expression>,
        fallback: Box<Expression>,
    },
    Fallback {
        value: Box<Expression>,
        fallback: Box<Expression>,
    },
    Throw {
        error: Box<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}
