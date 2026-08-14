use super::*;

impl Parser<'_> {
    pub(super) fn take_assign_op(&mut self) -> Option<AssignOp> {
        if self.take_simple(&TokenKind::Equal).is_some() {
            Some(AssignOp::Assign)
        } else if self.take_simple(&TokenKind::AddEqual).is_some() {
            Some(AssignOp::AddAssign)
        } else if self.take_simple(&TokenKind::SubEqual).is_some() {
            Some(AssignOp::SubAssign)
        } else if self.take_simple(&TokenKind::MulEqual).is_some() {
            Some(AssignOp::MulAssign)
        } else if self.take_simple(&TokenKind::DivEqual).is_some() {
            Some(AssignOp::DivAssign)
        } else if self.take_simple(&TokenKind::ModEqual).is_some() {
            Some(AssignOp::ModAssign)
        } else {
            None
        }
    }

    pub(super) fn expect_identifier(&mut self, expected: &str) -> Result<Ident, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Identifier(name) => Ok(Ident {
                span: token.span,
                name,
            }),
            // `send` is a statement keyword only in expression-leading
            // position. It remains available as a library/member API name,
            // such as `network.send(...)`.
            TokenKind::Send => Ok(Ident {
                span: token.span,
                name: "send".into(),
            }),
            // `view` is an ownership operator only in expression-leading
            // position. Tensor APIs may still expose `model.view(...)`.
            TokenKind::View => Ok(Ident {
                span: token.span,
                name: "view".into(),
            }),
            // `from` is contextual after `def` and `.`, where it names the
            // language conversion hook rather than beginning an import.
            TokenKind::From => Ok(Ident {
                span: token.span,
                name: "from".into(),
            }),
            _ => Err(ParseError {
                span: token.span,
                message: format!("expected {expected}"),
            }),
        }
    }

    pub(super) fn expect_member_name(&mut self) -> Result<Ident, ParseError> {
        if self.at(&TokenKind::From) || self.at(&TokenKind::With) {
            let token = self.advance().clone();
            let name = match token.kind {
                TokenKind::From => "from",
                TokenKind::With => "with",
                _ => unreachable!(),
            };
            return Ok(Ident {
                span: token.span,
                name: name.into(),
            });
        }
        self.expect_identifier("member name")
    }

    pub(super) fn expect_simple(
        &mut self,
        kind: TokenKind,
        expected: &str,
    ) -> Result<Token, ParseError> {
        if self.at(&kind) {
            Ok(self.advance().clone())
        } else {
            Err(self.error(&format!("expected {expected}")))
        }
    }

    pub(super) fn take_simple(&mut self, kind: &TokenKind) -> Option<&Token> {
        if self.at(kind) {
            let index = self.current;
            self.current += 1;
            Some(&self.tokens[index])
        } else {
            None
        }
    }

    pub(super) fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    pub(super) fn peek_kind(&self, offset: usize, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek_token(offset).kind) == std::mem::discriminant(kind)
    }

    pub(super) fn peek_token(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.current + offset)
            .unwrap_or_else(|| self.tokens.last().unwrap())
    }

    pub(super) fn advance(&mut self) -> &Token {
        let index = self.current;
        if !matches!(self.tokens[index].kind, TokenKind::Eof) {
            self.current += 1;
        }
        &self.tokens[index]
    }

    pub(super) fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    pub(super) fn error(&self, message: &str) -> ParseError {
        ParseError {
            span: self.peek().span,
            message: message.into(),
        }
    }
}

pub(super) fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
    Expr::Binary(BinaryExpr {
        span: Span::new(left.span().start, right.span().end),
        left: Box::new(left),
        op,
        right: Box::new(right),
    })
}

pub(super) fn normalize_quantity(value: i64, unit: &str, span: Span) -> Result<i64, ParseError> {
    let multiplier = match unit {
        "ns" => 1,
        "us" => 1_000,
        "ms" => 1_000_000,
        "s" => 1_000_000_000,
        "b" => 1,
        "kb" => 1024,
        "mb" => 1024 * 1024,
        "gb" => 1024 * 1024 * 1024,
        _ => {
            return Err(ParseError {
                span,
                message: format!(
                    "unknown contract quantity unit `{unit}`; use ns, us, ms, s, b, kb, mb, or gb"
                ),
            })
        }
    };
    value.checked_mul(multiplier).ok_or_else(|| ParseError {
        span,
        message: format!("contract quantity `{value}{unit}` is too large"),
    })
}

pub(super) fn task_placement(name: &str) -> Option<TaskPlacement> {
    match name {
        "local" => Some(TaskPlacement::Local),
        "gpu" => Some(TaskPlacement::Gpu),
        "simd" => Some(TaskPlacement::Simd),
        "simt" => Some(TaskPlacement::Simt),
        _ => None,
    }
}

pub(super) fn is_task_context_symbol(name: &str) -> bool {
    matches!(name, "self" | "runtime") || task_placement(name).is_some()
}

pub(super) fn internal_call(name: &str, span: Span, args: Vec<Expr>) -> Expr {
    Expr::Call(CallExpr {
        span,
        callee: Box::new(Expr::Identifier(Ident {
            span,
            name: name.into(),
        })),
        args: args
            .into_iter()
            .map(|value| CallArg {
                span: value.span(),
                name: None,
                value,
            })
            .collect(),
    })
}
