use crate::TypeContext;
use std::fmt;

#[derive(Debug, Clone)]
pub struct UniversalContext {
    pub types: TypeContext,
}

impl UniversalContext {
    pub fn new(types: TypeContext) -> Self {
        Self { types }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniversalError {
    Message(String),
}

impl fmt::Display for UniversalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for UniversalError {}
