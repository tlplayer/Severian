use crate::{BindingId, DefId, HirId, OpId, Substitution, TypeId, VariantId};
use severian_source::Span;
use severian_universal::{BinaryOperator, LiteralValue, UnaryOperator};

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
    Convert {
        operand: Box<Expression>,
        conversion: Conversion,
    },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    pub from: TypeId,
    pub to: TypeId,
    pub kind: ConversionKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionKind {
    NumericWidening,
    UnionInjection,
    Borrow,
    User(DefId),
}
