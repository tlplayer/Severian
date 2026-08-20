use crate::{TargetSpec, TypeContext};
use std::fmt;

#[derive(Debug, Clone)]
pub struct UniversalContext {
    pub types: TypeContext,
    pub target: TargetSpec,
}

impl UniversalContext {
    pub fn new(types: TypeContext, target: TargetSpec) -> Self {
        Self { types, target }
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
