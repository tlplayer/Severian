use severian_ast::BinaryOperator;
use severian_source::Span;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MutationKind {
    BooleanLiteral,
    Comparison,
    Arithmetic,
    Logical,
    Conditional,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mutation {
    pub id: usize,
    pub kind: MutationKind,
    pub file: PathBuf,
    pub span: Span,
    pub original: String,
    pub replacement: String,
    pub(super) edit: MutationEdit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MutationEdit {
    Boolean {
        expression: Span,
        value: bool,
    },
    Binary {
        expression: Span,
        original: BinaryOperator,
        replacement: BinaryOperator,
    },
    NegateConditional {
        condition: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MutationStatus {
    Killed,
    Survived,
    CompileKilled,
    TimeoutKilled,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MutationResult {
    pub mutation: Mutation,
    pub status: MutationStatus,
}

impl MutationStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Killed => "KILLED",
            Self::Survived => "SURVIVED",
            Self::CompileKilled => "COMPILE",
            Self::TimeoutKilled => "TIMEOUT",
            Self::Skipped => "SKIPPED",
        }
    }

    pub const fn is_killed(self) -> bool {
        matches!(
            self,
            Self::Killed | Self::CompileKilled | Self::TimeoutKilled
        )
    }
}
