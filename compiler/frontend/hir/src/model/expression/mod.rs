use crate::{BindingId, DefId, HirId, OpId, Substitution, TypeId, VariantId};
use severian_source::Span;
use severian_universal::{BinaryOperator, Conversion, LiteralValue, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwner {
    SelfScope,
    Runtime,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expression {
    pub id: HirId,
    pub type_id: TypeId,
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Literal(LiteralValue),
    Binding(BindingId),
    Function(DefId),
    Aggregate {
        class: TypeId,
        fields: Vec<Expression>,
    },
    Field {
        object: Box<Expression>,
        index: u32,
    },
    Call {
        callee: Callee,
        arguments: Vec<Expression>,
    },
    Async {
        expression: Box<Expression>,
        owner: TaskOwner,
        locked: bool,
    },
    AsyncFieldUpdate {
        binding: BindingId,
        field: u32,
        operator: BinaryOperator,
        value: Box<Expression>,
        owner: TaskOwner,
        locked: bool,
    },
    Await(Box<Expression>),
    Fallback {
        condition: Box<Expression>,
        value: Box<Expression>,
        fallback: Box<Expression>,
    },
    Throw(Box<Expression>),
    Convert {
        operand: Box<Expression>,
        conversion: Conversion,
    },
    Borrow {
        operand: Box<Expression>,
        exclusive: bool,
    },
    Move(Box<Expression>),
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Callee {
    Direct {
        function: DefId,
        substitution: Substitution,
    },
    FunctionValue(HirId),
    Method {
        implementation: DefId,
        receiver: HirId,
        substitution: Substitution,
    },
    Constructor {
        type_def: DefId,
        variant: Option<VariantId>,
    },
    Intrinsic(OpId),
}
