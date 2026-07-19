#![forbid(unsafe_code)]

use severian_ast::{
    Block, CallArg, CallExpr, Expr, FunctionDecl, Ident, Item, Literal, Module, Span, Stmt,
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
        self.expect_simple(TokenKind::RightParen, "`)`")?;
        self.expect_simple(TokenKind::Colon, "`:`")?;
        self.expect_simple(TokenKind::Newline, "newline after function header")?;
        self.expect_simple(TokenKind::Indent, "indented function body")?;
        let body = self.parse_block()?;
        let end = self
            .expect_simple(TokenKind::Dedent, "end of function body")?
            .span
            .end;

        Ok(FunctionDecl {
            span: Span::new(start, end),
            decorators: Vec::new(),
            name,
            params: Vec::new(),
            return_type: None,
            body,
            tests: Vec::new(),
        })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start = self.peek().span.start;
        let mut statements = Vec::new();

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let expression = self.parse_expression()?;
            self.expect_simple(TokenKind::Newline, "newline after statement")?;
            statements.push(Stmt::Expr(expression));
        }

        if statements.is_empty() {
            return Err(ParseError {
                span: self.peek().span,
                message: "function body cannot be empty".into(),
            });
        }

        let end = statements.last().unwrap().span().end;
        Ok(Block {
            span: Span::new(start, end),
            statements,
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        let expression = self.parse_primary()?;

        if self.take_simple(&TokenKind::LeftParen).is_some() {
            let start = expression.span().start;
            let mut args = Vec::new();
            if !self.at(&TokenKind::RightParen) {
                loop {
                    let value = self.parse_primary()?;
                    let span = value.span();
                    args.push(CallArg {
                        span,
                        name: None,
                        value,
                    });
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            let end = self.expect_simple(TokenKind::RightParen, "`)`")?.span.end;
            Ok(Expr::Call(CallExpr {
                span: Span::new(start, end),
                callee: Box::new(expression),
                args,
            }))
        } else {
            Ok(expression)
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Identifier(name) => Ok(Expr::Identifier(Ident {
                span: token.span,
                name,
            })),
            TokenKind::String(value) => Ok(Expr::Literal(Literal::String {
                span: token.span,
                value,
            })),
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
            Err(ParseError {
                span: self.peek().span,
                message: format!("expected {expected}"),
            })
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
}
