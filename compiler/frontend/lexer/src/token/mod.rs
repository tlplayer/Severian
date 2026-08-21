use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Identifier(String),
    Integer(String),
    Float(String),
    Character(char),
    String(String),
    At,
    Colon,
    ColonEqual,
    Equal,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Arrow,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Pipe,
    Comma,
    Dot,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
