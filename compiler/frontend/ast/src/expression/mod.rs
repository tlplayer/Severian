use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Integer(String),
    Float(String),
    Boolean(bool),
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
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Literal(Literal),
    Name(String),
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
