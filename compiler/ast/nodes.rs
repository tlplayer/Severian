// compiler/ast/src/nodes.rs

#[derive(Debug, Clone)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Function(FunctionDecl),
    Class(ClassDecl),
    Trait(TraitDecl),
    Statement(Stmt),
}

//
// ===== Declarations =====
//

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: String,
    pub traits: Vec<String>,
    pub fields: Vec<Field>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub methods: Vec<TraitMethod>,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
}

//
// ===== Statements =====
//

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    Assign(AssignStmt),
    Return(Option<Expr>),
    If(IfStmt),
    While(WhileStmt),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub mutable: bool,
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct AssignStmt {
    pub target: Expr,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_block: Block,
    pub else_block: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Block,
}

//
// ===== Expressions =====
//

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Identifier(String),

    Binary(BinaryExpr),
    Unary(UnaryExpr),

    Call(CallExpr),

    Member(MemberExpr),

    List(Vec<Expr>),
    Tuple(Vec<Expr>),

    Index(IndexExpr),
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub left: Box<Expr>,
    pub op: BinaryOp,
    pub right: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub expr: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct CallExpr {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct MemberExpr {
    pub object: Box<Expr>,
    pub member: String,
}

#[derive(Debug, Clone)]
pub struct IndexExpr {
    pub object: Box<Expr>,
    pub index: Box<Expr>,
}

//
// ===== Literals =====
//

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Null,
}

//
// ===== Types =====
//

#[derive(Debug, Clone)]
pub enum Type {
    Named(String),
    List(Box<Type>),
    Tuple(Vec<Type>),
    Function {
        params: Vec<Type>,
        returns: Box<Type>,
    },
}

//
// ===== Operators =====
//

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    Equal,
    NotEqual,

    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    And,
    Or,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

pub struct BinaryExpr {
    pub span: Span,
    pub left: Box<Expr>,
    pub op: BinaryOp,
    pub right: Box<Expr>,
}