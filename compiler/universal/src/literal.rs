#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LiteralKind {
    Integer,
    Float,
    Boolean,
    Character,
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
    Character(char),
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
            Self::Character(_) => LiteralKind::Character,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_literal_reports_its_kind_and_only_textual_values_have_spellings() {
        let cases = [
            (LiteralValue::Integer("42".into()), LiteralKind::Integer, Some("42")),
            (LiteralValue::Float("3.5".into()), LiteralKind::Float, Some("3.5")),
            (LiteralValue::Boolean(true), LiteralKind::Boolean, None),
            (LiteralValue::Character('x'), LiteralKind::Character, None),
            (LiteralValue::String("text".into()), LiteralKind::String, Some("text")),
            (LiteralValue::Bytes(vec![1, 2]), LiteralKind::Bytes, None),
            (LiteralValue::None, LiteralKind::None, None),
            (LiteralValue::Unit, LiteralKind::Unit, None),
        ];

        for (literal, kind, spelling) in cases {
            assert_eq!(literal.kind(), kind);
            assert_eq!(literal.spelling(), spelling);
        }
    }
}
