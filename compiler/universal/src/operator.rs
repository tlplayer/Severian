use crate::{LiteralKind, TypeId};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinaryOperator {
    BitwiseOr,
    BitwiseAnd,
    BitwiseXor,
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
    Contains,
    And,
    Or,
}

impl BinaryOperator {
    pub fn from_spelling(value: &str) -> Option<Self> {
        match value {
            "|" => Some(Self::BitwiseOr),
            "&" => Some(Self::BitwiseAnd),
            "^" => Some(Self::BitwiseXor),
            "+" => Some(Self::Add),
            "-" => Some(Self::Subtract),
            "*" => Some(Self::Multiply),
            "/" => Some(Self::Divide),
            "%" => Some(Self::Remainder),
            "**" => Some(Self::Power),
            "==" => Some(Self::Equal),
            "!=" => Some(Self::NotEqual),
            "<" => Some(Self::Less),
            "<=" => Some(Self::LessEqual),
            ">" => Some(Self::Greater),
            ">=" => Some(Self::GreaterEqual),
            "in" => Some(Self::Contains),
            "and" => Some(Self::And),
            "or" => Some(Self::Or),
            _ => None,
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::BitwiseOr => "|",
            Self::BitwiseAnd => "&",
            Self::BitwiseXor => "^",
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::Power => "**",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Contains => "in",
            Self::And => "and",
            Self::Or => "or",
        })
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
        ] {
            assert_eq!(BinaryOperator::from_spelling(spelling), Some(operator));
            assert_eq!(operator.to_string(), spelling);
        }
    }
}
