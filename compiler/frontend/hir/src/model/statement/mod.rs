use crate::{BindingId, Expression, TypeId, VariableId};
use severian_source::Span;
use severian_universal::BinaryOperator;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub id: BindingId,
    pub variable: VariableId,
    pub type_id: TypeId,
    pub value: Expression,
    /// Whether the source variable was declared with `:=` and may therefore
    /// receive later assignments.
    pub mutable: bool,
    /// True for `?=`. False means a fallible value is unwrapped and its error
    /// path is propagated when result lowering is available.
    pub preserve_error: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Sequence(Block),
    Binding(BindingId),
    FieldUpdate {
        binding: BindingId,
        field: u32,
        operator: BinaryOperator,
        value: Expression,
    },
    FieldSet {
        binding: BindingId,
        field: u32,
        value: Expression,
    },
    Expression(Expression),
    Return(Option<Expression>),
    Assert {
        condition: Expression,
        message: Option<Expression>,
        span: Span,
        condition_span: Span,
    },
    If {
        condition: Expression,
        then_block: Block,
        else_block: Block,
    },
    While {
        condition: Expression,
        body: Block,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Match {
        subject: Expression,
        arms: Vec<MatchArm>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub binding: Option<BindingId>,
    pub type_id: Option<TypeId>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub statements: Vec<Statement>,
}
