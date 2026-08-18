use crate::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Binary(BinaryExpr),
    Unary(UnaryExpr),
    Call(CallExpr),
    Member(MemberExpr),
    List(CollectionExpr),
    ListComprehension(ListComprehensionExpr),
    SetComprehension(SetComprehensionExpr),
    MapComprehension(MapComprehensionExpr),
    Tuple(CollectionExpr),
    Map(MapExpr),
    Set(CollectionExpr),
    Index(IndexExpr),
    Slice(SliceExpr),
    If(IfExpr),
    Switch(SwitchExpr),
    Lambda(LambdaExpr),
    Await(AwaitExpr),
    Async(AsyncExpr),
    Channel(ChannelExpr),
    Send(SendExpr),
    Ownership(OwnershipExpr),
    ChaosRule(ChaosRuleExpr),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal(node) => node.span(),
            Expr::Identifier(node) => node.span,
            Expr::Binary(node) => node.span,
            Expr::Unary(node) => node.span,
            Expr::Call(node) => node.span,
            Expr::Member(node) => node.span,
            Expr::List(node) | Expr::Tuple(node) | Expr::Set(node) => node.span,
            Expr::ListComprehension(node) => node.span,
            Expr::SetComprehension(node) => node.span,
            Expr::MapComprehension(node) => node.span,
            Expr::Map(node) => node.span,
            Expr::Index(node) => node.span,
            Expr::Slice(node) => node.span,
            Expr::If(node) => node.span,
            Expr::Switch(node) => node.span,
            Expr::Lambda(node) => node.span,
            Expr::Await(node) => node.span,
            Expr::Async(node) => node.span,
            Expr::Channel(node) => node.span,
            Expr::Send(node) => node.span,
            Expr::Ownership(node) => node.span,
            Expr::ChaosRule(node) => node.span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpr {
    pub span: Span,
    pub left: Box<Expr>,
    pub op: BinaryOp,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpr {
    pub span: Span,
    pub op: UnaryOp,
    pub expr: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr {
    pub span: Span,
    pub callee: Box<Expr>,
    pub args: Vec<CallArg>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallArg {
    pub span: Span,
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemberExpr {
    pub span: Span,
    pub object: Box<Expr>,
    pub member: Ident,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionExpr {
    pub span: Span,
    pub elements: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ListComprehensionExpr {
    pub span: Span,
    pub element: Box<Expr>,
    pub clauses: Vec<ComprehensionClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SetComprehensionExpr {
    pub span: Span,
    pub element: Box<Expr>,
    pub clauses: Vec<ComprehensionClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapComprehensionExpr {
    pub span: Span,
    pub key: Box<Expr>,
    pub value: Box<Expr>,
    pub clauses: Vec<ComprehensionClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComprehensionClause {
    pub pattern: Pattern,
    pub iterable: Box<Expr>,
    pub condition: Option<Box<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapExpr {
    pub span: Span,
    pub entries: Vec<MapEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapEntry {
    pub span: Span,
    pub key: Expr,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexExpr {
    pub span: Span,
    pub object: Box<Expr>,
    pub index: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceExpr {
    pub span: Span,
    pub object: Box<Expr>,
    pub start: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
    pub step: Option<Box<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpr {
    pub span: Span,
    pub condition: Box<Expr>,
    pub then_expr: Box<Expr>,
    pub else_expr: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchExpr {
    pub span: Span,
    pub value: Box<Expr>,
    pub arms: Vec<SwitchExprArm>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchExprArm {
    pub span: Span,
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaExpr {
    pub span: Span,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: LambdaBody,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LambdaBody {
    Expr(Box<Expr>),
    Block(Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AwaitExpr {
    pub span: Span,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsyncExpr {
    pub span: Span,
    pub value: Box<Expr>,
    pub owner: TaskOwner,
    pub placement: TaskPlacement,
    pub captures: Vec<Ident>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwner {
    SelfOwned,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPlacement {
    Default,
    Local,
    Gpu,
    Simd,
    Simt,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelExpr {
    pub span: Span,
    pub element_type: Type,
    pub capacity: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SendExpr {
    pub span: Span,
    pub value: Box<Expr>,
    pub channel: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipExpr {
    pub span: Span,
    pub value: Box<Expr>,
    pub op: OwnershipOp,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChaosRuleExpr {
    pub span: Span,
    pub function: Box<Expr>,
    pub action: ChaosAction,
    pub value: Box<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaosAction {
    Return,
    Throw,
}

//
// ===== Patterns =====
//
