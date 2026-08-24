use crate::{Expression, TypeAnnotation};
use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub annotation: Option<TypeAnnotation>,
    pub value: Expression,
    /// `:=` declares mutable storage. Bindings declared with `=` are
    /// immutable after initialization.
    pub mutable: bool,
    /// Compound assignment creates a new SSA binding for an existing source
    /// name instead of declaring a second lexical name.
    pub update: bool,
    /// `?=` keeps the complete result union instead of propagating its error
    /// member out of the current function.
    pub preserve_error: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchCase {
    pub binding: Option<String>,
    pub annotation: Option<TypeAnnotation>,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Binding(Binding),
    FieldAssignment {
        object: Expression,
        field: String,
        value: Expression,
        span: Span,
    },
    Expression(Expression),
    Return {
        value: Option<Expression>,
        span: Span,
    },
    Assert {
        condition: Expression,
        message: Option<Expression>,
        span: Span,
    },
    Unsafe {
        body: Vec<Statement>,
        span: Span,
    },
    Try {
        body: Vec<Statement>,
        catch_binding: String,
        catch_annotation: Option<TypeAnnotation>,
        catch_body: Vec<Statement>,
        span: Span,
    },
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Vec<Statement>,
        span: Span,
    },
    While {
        condition: Expression,
        initializer: Option<Binding>,
        body: Vec<Statement>,
        span: Span,
    },
    For {
        binding: String,
        second_binding: Option<String>,
        iterable: Expression,
        body: Vec<Statement>,
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
        cases: Vec<MatchCase>,
        span: Span,
    },
}
