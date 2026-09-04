use crate::{LiteralKind, OperationId, TypeId};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnaryOperator {
    Positive,
    Negative,
    Not,
}

impl UnaryOperator {
    pub fn from_spelling(value: &str) -> Option<Self> {
        match value {
            "+" => Some(Self::Positive),
            "-" => Some(Self::Negative),
            "not" => Some(Self::Not),
            _ => None,
        }
    }
}

/// The semantic identity of an operator.
///
/// This is intentionally an open stable ID, not an enum. Standard operators
/// are constants for convenience, while user declarations can construct an ID
/// from any symbol without changing the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperatorId(pub OperationId);

/// Compatibility name retained for the existing type-resolution API.
pub type BinaryOperator = OperatorId;

#[allow(non_upper_case_globals)]
impl OperatorId {
    pub const BitwiseOr: Self = Self::from_symbol("|");
    pub const BitwiseAnd: Self = Self::from_symbol("&");
    pub const BitwiseXor: Self = Self::from_symbol("^");
    pub const Add: Self = Self::from_symbol("+");
    pub const Subtract: Self = Self::from_symbol("-");
    pub const Multiply: Self = Self::from_symbol("*");
    pub const Divide: Self = Self::from_symbol("/");
    pub const FloorDivide: Self = Self::from_symbol("//");
    pub const Remainder: Self = Self::from_symbol("%");
    pub const Power: Self = Self::from_symbol("**");
    pub const ShiftLeft: Self = Self::from_symbol("<<");
    pub const ShiftRight: Self = Self::from_symbol(">>");
    pub const Equal: Self = Self::from_symbol("==");
    pub const NotEqual: Self = Self::from_symbol("!=");
    pub const Less: Self = Self::from_symbol("<");
    pub const LessEqual: Self = Self::from_symbol("<=");
    pub const Greater: Self = Self::from_symbol(">");
    pub const GreaterEqual: Self = Self::from_symbol(">=");
    pub const Contains: Self = Self::from_symbol("in");
    pub const And: Self = Self::from_symbol("and");
    pub const Or: Self = Self::from_symbol("or");

    pub const fn from_symbol(value: &str) -> Self {
        Self(OperationId::from_name(value))
    }

    pub const fn from_stable_id(value: u128) -> Self {
        Self(OperationId(value))
    }

    pub fn from_spelling(value: &str) -> Option<Self> {
        (!value.is_empty()).then(|| Self::from_symbol(value))
    }

    pub const fn standard_spelling(self) -> Option<&'static str> {
        if self.0 .0 == Self::BitwiseOr.0 .0 {
            Some("|")
        } else if self.0 .0 == Self::BitwiseAnd.0 .0 {
            Some("&")
        } else if self.0 .0 == Self::BitwiseXor.0 .0 {
            Some("^")
        } else if self.0 .0 == Self::Add.0 .0 {
            Some("+")
        } else if self.0 .0 == Self::Subtract.0 .0 {
            Some("-")
        } else if self.0 .0 == Self::Multiply.0 .0 {
            Some("*")
        } else if self.0 .0 == Self::Divide.0 .0 {
            Some("/")
        } else if self.0 .0 == Self::FloorDivide.0 .0 {
            Some("//")
        } else if self.0 .0 == Self::Remainder.0 .0 {
            Some("%")
        } else if self.0 .0 == Self::Power.0 .0 {
            Some("**")
        } else if self.0 .0 == Self::ShiftLeft.0 .0 {
            Some("<<")
        } else if self.0 .0 == Self::ShiftRight.0 .0 {
            Some(">>")
        } else if self.0 .0 == Self::Equal.0 .0 {
            Some("==")
        } else if self.0 .0 == Self::NotEqual.0 .0 {
            Some("!=")
        } else if self.0 .0 == Self::Less.0 .0 {
            Some("<")
        } else if self.0 .0 == Self::LessEqual.0 .0 {
            Some("<=")
        } else if self.0 .0 == Self::Greater.0 .0 {
            Some(">")
        } else if self.0 .0 == Self::GreaterEqual.0 .0 {
            Some(">=")
        } else if self.0 .0 == Self::Contains.0 .0 {
            Some("in")
        } else if self.0 .0 == Self::And.0 .0 {
            Some("and")
        } else if self.0 .0 == Self::Or.0 .0 {
            Some("or")
        } else {
            None
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(spelling) = self.standard_spelling() {
            formatter.write_str(spelling)
        } else {
            write!(formatter, "operator#{:032x}", self.0 .0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypePattern {
    Exact(TypeId),
    SameAsLeft,
    SameAsRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeConstraint {
    Known(TypeId),
    Literal(LiteralKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorSignature {
    pub operator: BinaryOperator,
    pub left: TypePattern,
    pub right: TypePattern,
    pub result: TypePattern,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitwise_spellings_are_first_class_universal_operators() {
        for (spelling, operator) in [
            ("|", BinaryOperator::BitwiseOr),
            ("&", BinaryOperator::BitwiseAnd),
            ("^", BinaryOperator::BitwiseXor),
            ("+", BinaryOperator::Add),
            ("-", BinaryOperator::Subtract),
            ("*", BinaryOperator::Multiply),
            ("/", BinaryOperator::Divide),
            ("%", BinaryOperator::Remainder),
            ("**", BinaryOperator::Power),
            ("==", BinaryOperator::Equal),
            ("!=", BinaryOperator::NotEqual),
            ("<", BinaryOperator::Less),
            ("<=", BinaryOperator::LessEqual),
            (">", BinaryOperator::Greater),
            (">=", BinaryOperator::GreaterEqual),
            ("in", BinaryOperator::Contains),
            ("and", BinaryOperator::And),
            ("or", BinaryOperator::Or),
        ] {
            assert_eq!(BinaryOperator::from_spelling(spelling), Some(operator));
            assert_eq!(operator.to_string(), spelling);
        }
        assert_eq!(BinaryOperator::from_spelling("??"), None);
    }

    #[test]
    fn unary_spellings_are_total_for_declared_operators() {
        assert_eq!(
            UnaryOperator::from_spelling("+"),
            Some(UnaryOperator::Positive)
        );
        assert_eq!(
            UnaryOperator::from_spelling("-"),
            Some(UnaryOperator::Negative)
        );
        assert_eq!(
            UnaryOperator::from_spelling("not"),
            Some(UnaryOperator::Not)
        );
        assert_eq!(UnaryOperator::from_spelling("!"), None);
    }
}
