// compiler/ast/nodes.rs

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(FunctionDecl),
    Class(ClassDecl),
    Trait(TraitDecl),
    Import(ImportDecl),
    Statement(Stmt),
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Function(node) => node.span,
            Item::Class(node) => node.span,
            Item::Trait(node) => node.span,
            Item::Import(node) => node.span,
            Item::Statement(node) => node.span(),
        }
    }
}

//
// ===== Source locations =====
//

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn empty(at: usize) -> Self {
        Self { start: at, end: at }
    }

    pub const fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident {
    pub span: Span,
    pub name: String,
}

//
// ===== Declarations =====
//

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub span: Span,
    pub kind: ImportKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportKind {
    Module {
        path: Vec<Ident>,
        alias: Option<Ident>,
    },
    From {
        module: Vec<Ident>,
        names: Vec<ImportName>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportName {
    pub span: Span,
    pub name: Ident,
    pub alias: Option<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub span: Span,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub tests: Vec<TestBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decorator {
    pub span: Span,
    pub name: TypePath,
    pub args: Vec<CallArg>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub span: Span,
    pub name: Ident,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub span: Span,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub traits: Vec<Type>,
    pub fields: Vec<Field>,
    pub constructors: Vec<ConstructorDecl>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub span: Span,
    pub name: Ident,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitDecl {
    pub span: Span,
    pub name: Ident,
    pub methods: Vec<TraitMethod>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstructorDecl {
    pub span: Span,
    pub decorators: Vec<Decorator>,
    pub name: Ident,
    pub params: Vec<Parameter>,
    pub body: Block,
    pub tests: Vec<TestBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestBlock {
    pub span: Span,
    pub name: Option<Ident>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub span: Span,
    pub name: Ident,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
}

//
// ===== Statements =====
//

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub span: Span,
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let(LetStmt),
    Assign(AssignStmt),
    Assert(AssertStmt),
    TryBind(TryBindStmt),
    Return(ReturnStmt),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Switch(SwitchStmt),
    Unsafe(UnsafeBlock),
    Expr(Expr),
    Break(Span),
    Continue(Span),
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let(node) => node.span,
            Stmt::Assign(node) => node.span,
            Stmt::Assert(node) => node.span,
            Stmt::TryBind(node) => node.span,
            Stmt::Return(node) => node.span,
            Stmt::If(node) => node.span,
            Stmt::While(node) => node.span,
            Stmt::For(node) => node.span,
            Stmt::Switch(node) => node.span,
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
    pub condition: Expr,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub span: Span,
    pub pattern: Pattern,
    pub iterable: Expr,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchStmt {
    pub span: Span,
    pub value: Expr,
    pub arms: Vec<SwitchArm>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchArm {
    pub span: Span,
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnsafeBlock {
    pub span: Span,
    pub body: Block,
}

//
// ===== Expressions =====
//

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Binary(BinaryExpr),
    Unary(UnaryExpr),
    Call(CallExpr),
    Member(MemberExpr),
    List(CollectionExpr),
    Tuple(CollectionExpr),
    Map(MapExpr),
    Set(CollectionExpr),
    Index(IndexExpr),
    If(IfExpr),
    Switch(SwitchExpr),
    Lambda(LambdaExpr),
    Await(AwaitExpr),
    Async(AsyncExpr),
    Ownership(OwnershipExpr),
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
            Expr::Map(node) => node.span,
            Expr::Index(node) => node.span,
            Expr::If(node) => node.span,
            Expr::Switch(node) => node.span,
            Expr::Lambda(node) => node.span,
            Expr::Await(node) => node.span,
            Expr::Async(node) => node.span,
            Expr::Ownership(node) => node.span,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipExpr {
    pub span: Span,
    pub value: Box<Expr>,
    pub op: OwnershipOp,
}

//
// ===== Patterns =====
//

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard(Span),
    Literal(Literal),
    Identifier(Ident),
    Tuple {
        span: Span,
        elements: Vec<Pattern>,
    },
    List {
        span: Span,
        elements: Vec<Pattern>,
    },
    Constructor {
        span: Span,
        name: Type,
        fields: Vec<Pattern>,
    },
    Or {
        span: Span,
        alternatives: Vec<Pattern>,
    },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Wildcard(span) => *span,
            Pattern::Literal(node) => node.span(),
            Pattern::Identifier(node) => node.span,
            Pattern::Tuple { span, .. }
            | Pattern::List { span, .. }
            | Pattern::Constructor { span, .. }
            | Pattern::Or { span, .. } => *span,
        }
    }
}

//
// ===== Literals =====
//

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer { span: Span, value: i64 },
    Float { span: Span, value: f64 },
    Boolean { span: Span, value: bool },
    String { span: Span, value: String },
    Null { span: Span },
}

impl Literal {
    pub fn span(&self) -> Span {
        match self {
            Literal::Integer { span, .. }
            | Literal::Float { span, .. }
            | Literal::Boolean { span, .. }
            | Literal::String { span, .. }
            | Literal::Null { span } => *span,
        }
    }
}

//
// ===== Types =====
//

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Named(TypePath),
    List {
        span: Span,
        element: Box<Type>,
    },
    Tuple {
        span: Span,
        elements: Vec<Type>,
    },
    Union {
        span: Span,
        alternatives: Vec<Type>,
    },
    Map {
        span: Span,
        key: Box<Type>,
        value: Box<Type>,
    },
    Set {
        span: Span,
        element: Box<Type>,
    },
    Result {
        span: Span,
        ok: Box<Type>,
        err: Box<Type>,
    },
    Option {
        span: Span,
        some: Box<Type>,
    },
    Function {
        span: Span,
        params: Vec<Type>,
        returns: Box<Type>,
    },
    Future {
        span: Span,
        output: Box<Type>,
    },
    Reference {
        span: Span,
        mutable: bool,
        inner: Box<Type>,
    },
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Named(node) => node.span,
            Type::List { span, .. }
            | Type::Tuple { span, .. }
            | Type::Union { span, .. }
            | Type::Map { span, .. }
            | Type::Set { span, .. }
            | Type::Result { span, .. }
            | Type::Option { span, .. }
            | Type::Function { span, .. }
            | Type::Future { span, .. }
            | Type::Reference { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypePath {
    pub span: Span,
    pub segments: Vec<Ident>,
    pub args: Vec<TypeArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeArg {
    pub span: Span,
    pub ty: Box<Type>,
}

//
// ===== Operators =====
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    MatMul,
    Cross,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipOp {
    View,
    Borrow,
    Clone,
    Move,
}
