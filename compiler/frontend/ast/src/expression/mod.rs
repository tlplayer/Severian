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
    Copy,
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
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
    pub value: Expression,
    /// The error shape expected by `throws(value -> ErrorType)`.
    pub expected_error: Option<TypeAnnotation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapEntry {
    pub key: Expression,
    pub value: Expression,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Literal(Literal),
    List(Vec<Expression>),
    Map(Vec<MapEntry>),
    Tuple(Vec<Expression>),
    Name(String),
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
