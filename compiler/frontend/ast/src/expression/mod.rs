use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Integer(i64),
    Name(String),
    Add {
        left: Box<Expression>,
        right: Box<Expression>,
    },
}
