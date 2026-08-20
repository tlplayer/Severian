#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LiteralKind {
    Integer,
    Float,
    Boolean,
    String,
    Bytes,
    None,
    Unit,
}

/// Literal spelling survives parsing until semantic constraints select a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiteralValue {
    Integer(String),
    Float(String),
    Boolean(bool),
    String(String),
    Bytes(Vec<u8>),
    None,
    Unit,
}

impl LiteralValue {
    pub const fn kind(&self) -> LiteralKind {
        match self {
            Self::Integer(_) => LiteralKind::Integer,
            Self::Float(_) => LiteralKind::Float,
            Self::Boolean(_) => LiteralKind::Boolean,
            Self::String(_) => LiteralKind::String,
            Self::Bytes(_) => LiteralKind::Bytes,
            Self::None => LiteralKind::None,
            Self::Unit => LiteralKind::Unit,
        }
    }

    pub fn spelling(&self) -> Option<&str> {
        match self {
            Self::Integer(spelling) | Self::Float(spelling) | Self::String(spelling) => {
                Some(spelling)
            }
            _ => None,
        }
    }
}
