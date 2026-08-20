use severian_ast::{Binding, Expression, ExpressionKind, Module};
use severian_diagnostics::Diagnostic;
use severian_lexer::{Token, TokenKind};
use severian_source::Span;

pub fn parse(tokens: &[Token]) -> Result<Module, Diagnostic> {
    Parser { tokens, cursor: 0 }.module()
}

struct Parser<'a> {
    tokens: &'a [Token],
    cursor: usize,
}

impl Parser<'_> {
    fn module(mut self) -> Result<Module, Diagnostic> {
        let mut bindings = Vec::new();
        self.separators();
        while !matches!(self.peek().kind, TokenKind::Eof) {
            bindings.push(self.binding()?);
            if !matches!(self.peek().kind, TokenKind::Separator | TokenKind::Eof) {
                return Err(self.error("expected a newline or comma after binding"));
            }
            self.separators();
        }
        Ok(Module { bindings })
    }

    fn binding(&mut self) -> Result<Binding, Diagnostic> {
        let name_token = self.next();
        let TokenKind::Identifier(name) = &name_token.kind else {
            return Err(Diagnostic::new(
                "E000110",
                "expected a binding name",
                Some(name_token.span),
            ));
        };
        let name = name.clone();
        self.expect(TokenKind::Equal, "expected `=` after binding name")?;
        let value = self.expression()?;
        Ok(Binding {
            name,
            span: Span::new(
                name_token.span.source,
                name_token.span.start,
                value.span.end,
            ),
            value,
        })
    }

    fn expression(&mut self) -> Result<Expression, Diagnostic> {
        let mut expression = self.primary()?;
        while matches!(self.peek().kind, TokenKind::Plus) {
            self.next();
            let right = self.primary()?;
            let span = Span::new(
                expression.span.source,
                expression.span.start,
                right.span.end,
            );
            expression = Expression {
                kind: ExpressionKind::Add {
                    left: Box::new(expression),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expression)
    }

    fn primary(&mut self) -> Result<Expression, Diagnostic> {
        let token = self.next();
        let kind = match &token.kind {
            TokenKind::Integer(value) => ExpressionKind::Integer(*value),
            TokenKind::Identifier(name) => ExpressionKind::Name(name.clone()),
            _ => {
                return Err(Diagnostic::new(
                    "E000111",
                    "expected an integer or binding name",
                    Some(token.span),
                ))
            }
        };
        Ok(Expression {
            kind,
            span: token.span,
        })
    }

    fn separators(&mut self) {
        while matches!(self.peek().kind, TokenKind::Separator) {
            self.cursor += 1;
        }
    }
    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }
    fn next(&mut self) -> Token {
        let token = self.tokens[self.cursor].clone();
        self.cursor += 1;
        token
    }
    fn expect(&mut self, expected: TokenKind, message: &str) -> Result<(), Diagnostic> {
        if std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(&expected) {
            self.cursor += 1;
            Ok(())
        } else {
            Err(self.error(message))
        }
    }
    fn error(&self, message: &str) -> Diagnostic {
        Diagnostic::new("E000112", message, Some(self.peek().span))
    }
}
