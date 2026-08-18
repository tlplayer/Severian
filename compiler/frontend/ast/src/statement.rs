use crate::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub span: Span,
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Function(FunctionDecl),
    Let(LetStmt),
    DestructureLet(DestructureLetStmt),
    Assign(AssignStmt),
    Assert(AssertStmt),
    TryBind(TryBindStmt),
    Return(ReturnStmt),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Switch(SwitchStmt),
    With(WithBlock),
    Unsafe(UnsafeBlock),
    Expr(Expr),
    Break(Span),
    Continue(Span),
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Function(node) => node.span,
            Stmt::Let(node) => node.span,
            Stmt::DestructureLet(node) => node.span,
            Stmt::Assign(node) => node.span,
            Stmt::Assert(node) => node.span,
            Stmt::TryBind(node) => node.span,
            Stmt::Return(node) => node.span,
            Stmt::If(node) => node.span,
            Stmt::While(node) => node.span,
            Stmt::For(node) => node.span,
            Stmt::Switch(node) => node.span,
            Stmt::With(node) => node.span,
            Stmt::Unsafe(node) => node.span,
            Stmt::Expr(node) => node.span(),
            Stmt::Break(span) | Stmt::Continue(span) => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetStmt {
    pub span: Span,
    pub kind: LetKind,
    pub name: Ident,
    pub ty: Option<Type>,
    pub value: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DestructureLetStmt {
    pub span: Span,
    pub names: Vec<Ident>,
    pub value: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LetKind {
    Stable,
    Changeable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignStmt {
    pub span: Span,
    pub target: Expr,
    pub op: AssignOp,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssertStmt {
    pub span: Span,
    pub condition: Expr,
    pub message: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryBindStmt {
    pub span: Span,
    pub name: Ident,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnStmt {
    pub span: Span,
    pub value: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfStmt {
    pub span: Span,
    pub condition: Expr,
    pub then_block: Block,
    pub else_branch: Option<ElseBranch>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElseBranch {
    If(Box<IfStmt>),
    Block(Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileStmt {
    pub span: Span,
    pub setup: Option<Box<Stmt>>,
    pub capabilities: Vec<Expr>,
    pub condition: Expr,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub span: Span,
    pub setup: Option<Box<Stmt>>,
    pub pattern: Pattern,
    pub iterable: Expr,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchStmt {
    pub span: Span,
    pub values: Vec<Expr>,
    pub repeat_condition: Option<Expr>,
    pub setup: Option<Box<Stmt>>,
    pub arms: Vec<SwitchArm>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchArm {
    pub span: Span,
    pub source: Option<Expr>,
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnsafeBlock {
    pub span: Span,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WithBlock {
    pub span: Span,
    pub resources: Vec<Expr>,
    pub body: Block,
}

//
// ===== Expressions =====
//
