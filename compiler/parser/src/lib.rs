#![forbid(unsafe_code)]

use severian_ast::{
    AssertStmt, AssignOp, AssignStmt, AsyncExpr, AwaitExpr, BinaryExpr, BinaryOp, Block, CallArg,
    CallExpr, ChaosAction, ChaosRuleExpr, ClassDecl, CollectionExpr, ComprehensionClause,
    ConstructorDecl, ContractClause, ContractFailure, Decorator, DecoratorSymbol,
    DestructureLetStmt, ElseBranch, EnumDecl, EnumVariant, Expr, Field, ForStmt, FunctionContract,
    FunctionDecl, GenericParameter, Ident, IfExpr, IfStmt, ImportDecl, ImportKind, ImportName,
    IndexExpr, Item, LambdaBody, LetKind, LetStmt, ListComprehensionExpr, Literal,
    MapComprehensionExpr, MapEntry, MapExpr, MemberExpr, Module, OwnershipExpr, OwnershipOp,
    Parameter, Pattern, ReturnStmt, SetComprehensionExpr, SliceExpr, Span, Stmt, SwitchArm,
    SwitchStmt, TaskOwner, TaskPlacement, TestBlock, TestMode, TraitDecl, TraitMethod, Type,
    TypeArg, TypePath, UnaryExpr, UnaryOp, UnsafeBlock, WhileStmt, WithBlock,
};
use severian_lexer::{Token, TokenKind};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}

pub fn parse(tokens: &[Token]) -> Result<Module, ParseError> {
    Parser {
        tokens,
        current: 0,
        test_depth: 0,
        unsafe_depth: 0,
        loop_depth: 0,
        task_contexts: Vec::new(),
    }
    .parse_module()
}

struct Parser<'tokens> {
    tokens: &'tokens [Token],
    current: usize,
    test_depth: usize,
    unsafe_depth: usize,
    loop_depth: usize,
    task_contexts: Vec<TaskContext>,
}

#[derive(Clone)]
struct TaskContext {
    owner: TaskOwner,
    placement: TaskPlacement,
    captures: Vec<Ident>,
}

mod cursor;
mod declaration;
mod expression;
mod function;
mod statement;
mod types;
