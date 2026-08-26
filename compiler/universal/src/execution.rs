use crate::AttributeId;
use std::fmt;

/// Source-level execution intent. This survives semantic lowering and is
/// consumed by CompileType handlers; it is not a promise that every backend
/// exists on every host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPlacement {
    Host,
    Simd,
    Gpu,
}

impl ExecutionPlacement {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "host" | "self" => Some(Self::Host),
            "simd" => Some(Self::Simd),
            "gpu" | "simt" => Some(Self::Gpu),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Simd => "simd",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for ExecutionPlacement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Placement requested by the source program.
pub const EXECUTION_PLACEMENT_ATTRIBUTE: AttributeId =
    AttributeId::from_name("execution.placement");

/// Concrete backend selected by the compiler driver.
pub const EXECUTION_BACKEND_ATTRIBUTE: AttributeId = AttributeId::from_name("execution.backend");

/// Concrete device selected by the compiler driver, when one is required.
pub const EXECUTION_DEVICE_ATTRIBUTE: AttributeId = AttributeId::from_name("execution.device");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_aliases_do_not_create_new_execution_kinds() {
        assert_eq!(
            ExecutionPlacement::parse("simt"),
            Some(ExecutionPlacement::Gpu)
        );
        assert_eq!(
            ExecutionPlacement::parse("simd"),
            Some(ExecutionPlacement::Simd)
        );
        assert_eq!(ExecutionPlacement::parse("distributed"), None);
    }
}
