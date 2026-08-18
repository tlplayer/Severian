use crate::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard(Span),
    Literal(Literal),
    Identifier(Ident),
    Tuple {
        span: Span,
        elements: Vec<Pattern>,
    },
    List {
        span: Span,
        elements: Vec<Pattern>,
    },
    Constructor {
        span: Span,
        name: Type,
        fields: Vec<Pattern>,
    },
    Or {
        span: Span,
        alternatives: Vec<Pattern>,
    },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Wildcard(span) => *span,
            Pattern::Literal(node) => node.span(),
            Pattern::Identifier(node) => node.span,
            Pattern::Tuple { span, .. }
            | Pattern::List { span, .. }
            | Pattern::Constructor { span, .. }
            | Pattern::Or { span, .. } => *span,
        }
    }
}

//
// ===== Literals =====
//

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer { span: Span, value: i64 },
    Float { span: Span, value: f64 },
    Boolean { span: Span, value: bool },
    String { span: Span, value: String },
    Null { span: Span },
}

impl Literal {
    pub fn span(&self) -> Span {
        match self {
            Literal::Integer { span, .. }
            | Literal::Float { span, .. }
            | Literal::Boolean { span, .. }
            | Literal::String { span, .. }
            | Literal::Null { span } => *span,
        }
    }
}

//
// ===== Types =====
//
