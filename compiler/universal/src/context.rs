use crate::{OperationRegistry, TypeContext};
use std::fmt;

#[derive(Debug, Clone)]
pub struct UniversalContext {
    pub types: TypeContext,
    pub operations: OperationRegistry,
}

impl UniversalContext {
    pub fn new(types: TypeContext) -> Self {
        Self {
            types,
            operations: OperationRegistry::default(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_starts_with_the_supplied_types_and_an_empty_registry() {
        let types = TypeContext::default();
        let context = UniversalContext::new(types);
        assert_eq!(context.types.definitions().count(), 0);
        assert_eq!(format!("{:?}", context.operations), "OperationRegistry { interfaces: 0 }");
    }

    #[test]
    fn universal_errors_preserve_their_message() {
        let error = UniversalError::Message("broken contract".into());
        assert_eq!(error.to_string(), "broken contract");
        assert_eq!(format!("{error:?}"), "Message(\"broken contract\")");
    }
}
