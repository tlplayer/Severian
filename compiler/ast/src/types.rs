use crate::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Named(TypePath),
    List {
        span: Span,
        element: Box<Type>,
    },
    Tuple {
        span: Span,
        elements: Vec<Type>,
    },
    Union {
        span: Span,
        alternatives: Vec<Type>,
    },
    Map {
        span: Span,
        key: Box<Type>,
        value: Box<Type>,
    },
    Set {
        span: Span,
        element: Box<Type>,
    },
    Result {
        span: Span,
        ok: Box<Type>,
        err: Box<Type>,
    },
    Option {
        span: Span,
        some: Box<Type>,
    },
    Function {
        span: Span,
        params: Vec<Type>,
        returns: Box<Type>,
    },
    Future {
        span: Span,
        output: Box<Type>,
    },
    Reference {
        span: Span,
        mutable: bool,
        inner: Box<Type>,
    },
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Named(node) => node.span,
            Type::List { span, .. }
            | Type::Tuple { span, .. }
            | Type::Union { span, .. }
            | Type::Map { span, .. }
            | Type::Set { span, .. }
            | Type::Result { span, .. }
            | Type::Option { span, .. }
            | Type::Function { span, .. }
            | Type::Future { span, .. }
            | Type::Reference { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypePath {
    pub span: Span,
    pub segments: Vec<Ident>,
    pub args: Vec<TypeArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeArg {
    Type { span: Span, ty: Box<Type> },
    Dimension { span: Span, size: u64 },
}

impl TypeArg {
    pub fn span(&self) -> Span {
        match self {
            Self::Type { span, .. } | Self::Dimension { span, .. } => *span,
        }
    }

    pub fn as_type(&self) -> Option<&Type> {
        match self {
            Self::Type { ty, .. } => Some(ty),
            Self::Dimension { .. } => None,
        }
    }
}

//
// ===== Operators =====
//
