#![forbid(unsafe_code)]

use severian_ast::{
    AssertStmt, BinaryExpr, BinaryOp, Block, CallArg, CallExpr, ElseBranch, Expr, FunctionDecl,
    Ident, IfStmt, Item, LetKind, LetStmt, Literal, Module, Parameter, ReturnStmt, Span, Stmt,
    TestBlock, Type, TypePath,
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
    Parser { tokens, current: 0 }.parse_module()
}

struct Parser<'tokens> {
    tokens: &'tokens [Token],
    current: usize,
}

impl Parser<'_> {
    fn parse_module(mut self) -> Result<Module, ParseError> {
        let start = self.peek().span.start;
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            if !self.at(&TokenKind::Def) {
                return Err(self.error("expected a function declaration"));
            }
            items.push(Item::Function(self.parse_function()?));
        }
        Ok(Module {
            span: Span::new(start, self.peek().span.end),
            items,
        })
    }

    fn parse_function(&mut self) -> Result<FunctionDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Def, "`def`")?.span.start;
        let name = self.expect_identifier("function name")?;
        self.expect_simple(TokenKind::LeftParen, "`(`")?;
        let mut params = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let param_start = self.peek().span.start;
                let param_name = self.expect_identifier("parameter name")?;
                self.expect_simple(TokenKind::Colon, "`:` after parameter name")?;
                let ty = self.parse_type()?;
                params.push(Parameter {
                    span: Span::new(param_start, ty.span().end),
                    name: param_name,
                    ty: Some(ty),
                    default: None,
                });
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect_simple(TokenKind::RightParen, "`)`")?;
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect_simple(TokenKind::Colon, "`:`")?;
        self.expect_simple(TokenKind::Newline, "newline after function header")?;
        self.expect_simple(TokenKind::Indent, "indented function body")?;
        let body = self.parse_block()?;
        let mut end = self
            .expect_simple(TokenKind::Dedent, "end of function body")?
            .span
            .end;

        let mut tests = Vec::new();
        while self.at(&TokenKind::Test) {
            let test = self.parse_test()?;
            end = test.span.end;
            tests.push(test);
        }

        Ok(FunctionDecl {
            span: Span::new(start, end),
            decorators: Vec::new(),
            name,
            params,
            return_type,
            body,
            tests,
        })
    }

    fn parse_test(&mut self) -> Result<TestBlock, ParseError> {
        let start = self.expect_simple(TokenKind::Test, "`test`")?.span.start;
        let name = if let TokenKind::String(value) = self.peek().kind.clone() {
            let token = self.advance().clone();
            Some(Ident {
                span: token.span,
                name: value,
            })
        } else {
            None
        };
        self.expect_simple(TokenKind::Colon, "`:` after test")?;
        self.expect_simple(TokenKind::Newline, "newline after test header")?;
        self.expect_simple(TokenKind::Indent, "indented test body")?;
        let body = self.parse_block()?;
        let end = self
            .expect_simple(TokenKind::Dedent, "end of test body")?
            .span
            .end;
        Ok(TestBlock {
            span: Span::new(start, end),
            name,
            body,
        })
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let name = self.expect_identifier("type name")?;
        Ok(Type::Named(TypePath {
            span: name.span,
            segments: vec![name],
            args: Vec::new(),
        }))
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start = self.peek().span.start;
        let mut statements = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            statements.push(self.parse_statement()?);
        }
        if statements.is_empty() {
            return Err(self.error("block cannot be empty"));
        }
        let end = statements.last().unwrap().span().end;
        Ok(Block {
            span: Span::new(start, end),
            statements,
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        if self.at(&TokenKind::If) {
            return self.parse_if().map(Stmt::If);
        }
        if self.at(&TokenKind::Return) {
            let start = self.advance().span.start;
            let value = if self.at(&TokenKind::Newline) {
                None
            } else {
                Some(self.parse_expression()?)
            };
            let end = self
                .expect_simple(TokenKind::Newline, "newline after return")?
                .span
                .end;
            return Ok(Stmt::Return(ReturnStmt {
                span: Span::new(start, end),
                value,
            }));
        }
        if self.at(&TokenKind::Assert) {
            let start = self.advance().span.start;
            let condition = self.parse_expression()?;
            let message = if self.take_simple(&TokenKind::Comma).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            let end = self
                .expect_simple(TokenKind::Newline, "newline after assertion")?
                .span
                .end;
            return Ok(Stmt::Assert(AssertStmt {
                span: Span::new(start, end),
                condition,
                message,
            }));
        }
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.peek_kind(1, &TokenKind::Equal)
        {
            let name = self.expect_identifier("binding name")?;
            let start = name.span.start;
            self.expect_simple(TokenKind::Equal, "`=`")?;
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after binding")?
                .span
                .end;
            return Ok(Stmt::Let(LetStmt {
                span: Span::new(start, end),
                kind: LetKind::Stable,
                name,
                ty: None,
                value: Some(value),
            }));
        }

        let expression = self.parse_expression()?;
        self.expect_simple(TokenKind::Newline, "newline after statement")?;
        Ok(Stmt::Expr(expression))
    }

    fn parse_if(&mut self) -> Result<IfStmt, ParseError> {
        let start = self.expect_simple(TokenKind::If, "`if`")?.span.start;
        let condition = self.parse_expression()?;
        self.expect_simple(TokenKind::Colon, "`:` after if condition")?;
        self.expect_simple(TokenKind::Newline, "newline after if header")?;
        self.expect_simple(TokenKind::Indent, "indented if body")?;
        let then_block = self.parse_block()?;
        let mut end = self
            .expect_simple(TokenKind::Dedent, "end of if body")?
            .span
            .end;
        let else_branch = if self.take_simple(&TokenKind::Else).is_some() {
            self.expect_simple(TokenKind::Colon, "`:` after else")?;
            self.expect_simple(TokenKind::Newline, "newline after else header")?;
            self.expect_simple(TokenKind::Indent, "indented else body")?;
            let block = self.parse_block()?;
            end = self
                .expect_simple(TokenKind::Dedent, "end of else body")?
                .span
                .end;
            Some(ElseBranch::Block(block))
        } else {
            None
        };
        Ok(IfStmt {
            span: Span::new(start, end),
            condition,
            then_block,
            else_branch,
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_equality()
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_comparison()?;
        loop {
            let op = if self.take_simple(&TokenKind::EqualEqual).is_some() {
                Some(BinaryOp::Equal)
            } else if self.take_simple(&TokenKind::NotEqual).is_some() {
                Some(BinaryOp::NotEqual)
            } else {
                None
            };
            let Some(op) = op else { break };
            expression = binary(expression, op, self.parse_comparison()?);
        }
        Ok(expression)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_addition()?;
        loop {
            let op = if self.take_simple(&TokenKind::Less).is_some() {
                Some(BinaryOp::Less)
            } else if self.take_simple(&TokenKind::LessEqual).is_some() {
                Some(BinaryOp::LessEqual)
            } else if self.take_simple(&TokenKind::Greater).is_some() {
                Some(BinaryOp::Greater)
            } else if self.take_simple(&TokenKind::GreaterEqual).is_some() {
                Some(BinaryOp::GreaterEqual)
            } else {
                None
            };
            let Some(op) = op else { break };
            expression = binary(expression, op, self.parse_addition()?);
        }
        Ok(expression)
    }

    fn parse_addition(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_multiplication()?;
        loop {
            let op = if self.take_simple(&TokenKind::Plus).is_some() {
                Some(BinaryOp::Add)
            } else if self.take_simple(&TokenKind::Minus).is_some() {
                Some(BinaryOp::Sub)
            } else {
                None
            };
            let Some(op) = op else { break };
            expression = binary(expression, op, self.parse_multiplication()?);
        }
        Ok(expression)
    }

    fn parse_multiplication(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_call()?;
        loop {
            let op = if self.take_simple(&TokenKind::Star).is_some() {
                Some(BinaryOp::Mul)
            } else if self.take_simple(&TokenKind::Slash).is_some() {
                Some(BinaryOp::Div)
            } else if self.take_simple(&TokenKind::Percent).is_some() {
                Some(BinaryOp::Mod)
            } else {
                None
            };
            let Some(op) = op else { break };
            expression = binary(expression, op, self.parse_call()?);
        }
        Ok(expression)
    }

    fn parse_call(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_primary()?;
        while self.take_simple(&TokenKind::LeftParen).is_some() {
            let start = expression.span().start;
            let mut args = Vec::new();
            if !self.at(&TokenKind::RightParen) {
                loop {
                    let value = self.parse_expression()?;
                    args.push(CallArg {
                        span: value.span(),
                        name: None,
                        value,
                    });
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            let end = self.expect_simple(TokenKind::RightParen, "`)`")?.span.end;
            expression = Expr::Call(CallExpr {
                span: Span::new(start, end),
                callee: Box::new(expression),
                args,
            });
        }
        Ok(expression)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Identifier(name) => Ok(Expr::Identifier(Ident {
                span: token.span,
                name,
            })),
            TokenKind::Integer(value) => Ok(Expr::Literal(Literal::Integer {
                span: token.span,
                value,
            })),
            TokenKind::String(value) => Ok(Expr::Literal(Literal::String {
                span: token.span,
                value,
            })),
            TokenKind::True => Ok(Expr::Literal(Literal::Boolean {
                span: token.span,
                value: true,
            })),
            TokenKind::False => Ok(Expr::Literal(Literal::Boolean {
                span: token.span,
                value: false,
            })),
            TokenKind::LeftParen => {
                let expression = self.parse_expression()?;
                self.expect_simple(TokenKind::RightParen, "`)`")?;
                Ok(expression)
            }
            _ => Err(ParseError {
                span: token.span,
                message: "expected an expression".into(),
            }),
        }
    }

    fn expect_identifier(&mut self, expected: &str) -> Result<Ident, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Identifier(name) => Ok(Ident {
                span: token.span,
                name,
            }),
            _ => Err(ParseError {
                span: token.span,
                message: format!("expected {expected}"),
            }),
        }
    }

    fn expect_simple(&mut self, kind: TokenKind, expected: &str) -> Result<Token, ParseError> {
        if self.at(&kind) {
            Ok(self.advance().clone())
        } else {
            Err(self.error(&format!("expected {expected}")))
        }
    }

    fn take_simple(&mut self, kind: &TokenKind) -> Option<&Token> {
        if self.at(kind) {
            let index = self.current;
            self.current += 1;
            Some(&self.tokens[index])
        } else {
            None
        }
    }

    fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    fn peek_kind(&self, offset: usize, kind: &TokenKind) -> bool {
        self.tokens.get(self.current + offset).is_some_and(|token| {
            std::mem::discriminant(&token.kind) == std::mem::discriminant(kind)
        })
    }

    fn advance(&mut self) -> &Token {
        let index = self.current;
        if !matches!(self.tokens[index].kind, TokenKind::Eof) {
            self.current += 1;
        }
        &self.tokens[index]
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn error(&self, message: &str) -> ParseError {
        ParseError {
            span: self.peek().span,
            message: message.into(),
        }
    }
}

fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
    Expr::Binary(BinaryExpr {
        span: Span::new(left.span().start, right.span().end),
        left: Box::new(left),
        op,
        right: Box::new(right),
    })
}
