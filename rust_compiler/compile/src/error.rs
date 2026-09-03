use severian_universal::{CompilerId, TypeId};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    ConflictingCompilers {
        operation: usize,
        compilers: Vec<CompilerId>,
    },
    InvalidArtifact(String),
    DuplicateHandler(CompilerId),
    MissingHandler(CompilerId),
    MissingValue(u32),
    PlannerGeneratedOperation(usize),
    CfgCompileType(CompilerId),
    Type(TypeId, String),
    Target(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingCompilers {
                operation,
                compilers,
            } => write!(
                formatter,
                "MIR operation {operation} mixes CompileTypes handled by {compilers:?}"
            ),
            Self::InvalidArtifact(message) => {
                write!(formatter, "invalid generated MLIR: {message}")
            }
            Self::DuplicateHandler(compiler) => {
                write!(
                    formatter,
                    "a CompileHandler is already registered for {compiler}"
                )
            }
            Self::MissingHandler(compiler) => {
                write!(formatter, "no CompileHandler is registered for {compiler}")
            }
            Self::MissingValue(value) => write!(formatter, "MIR references missing value %{value}"),
            Self::PlannerGeneratedOperation(operation) => write!(
                formatter,
                "MIR operation {operation} is a planner-generated region call"
            ),
            Self::CfgCompileType(compiler) => write!(
                formatter,
                "CompileType handler {compiler} cannot consume CFG MIR yet"
            ),
            Self::Type(ty, message) => write!(formatter, "cannot route type {ty:?}: {message}"),
            Self::Target(message) => write!(formatter, "cannot route compiled region: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}
