use super::*;
use crate::cursor::{binary, internal_call, normalize_quantity, task_placement};

impl Parser<'_> {
    pub(super) fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_conditional()
    }

    pub(super) fn parse_conditional(&mut self) -> Result<Expr, ParseError> {
        let then_expr = self.parse_or()?;
        if self.take_simple(&TokenKind::If).is_none() {
            return Ok(then_expr);
        }
        let condition = self.parse_or()?;
        self.expect_simple(TokenKind::Else, "`else` in conditional expression")?;
        let else_expr = self.parse_conditional()?;
        Ok(Expr::If(IfExpr {
            span: Span::new(then_expr.span().start, else_expr.span().end),
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }))
    }

    pub(super) fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_and()?;
        while self.take_simple(&TokenKind::Or).is_some() {
            expression = binary(expression, BinaryOp::Or, self.parse_and()?);
        }
        Ok(expression)
    }

    pub(super) fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_equality()?;
        while self.take_simple(&TokenKind::And).is_some() {
            expression = binary(expression, BinaryOp::And, self.parse_equality()?);
        }
        Ok(expression)
    }

    pub(super) fn parse_equality(&mut self) -> Result<Expr, ParseError> {
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

    pub(super) fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_addition()?;
        let mut chain: Option<Expr> = None;
        loop {
            let negated_membership = self.at(&TokenKind::Not) && self.peek_kind(1, &TokenKind::In);
            let op = if self.take_simple(&TokenKind::Less).is_some() {
                Some(BinaryOp::Less)
            } else if self.take_simple(&TokenKind::LessEqual).is_some() {
                Some(BinaryOp::LessEqual)
            } else if self.take_simple(&TokenKind::Greater).is_some() {
                Some(BinaryOp::Greater)
            } else if self.take_simple(&TokenKind::GreaterEqual).is_some() {
                Some(BinaryOp::GreaterEqual)
            } else if self.take_simple(&TokenKind::In).is_some() {
                Some(BinaryOp::In)
            } else if negated_membership {
                self.advance();
                self.advance();
                Some(BinaryOp::In)
            } else {
                None
            };
            let Some(op) = op else { break };
            let right = self.parse_addition()?;
            let comparison = binary(left, op, right.clone());
            let comparison = if negated_membership {
                let span = comparison.span();
                Expr::Unary(UnaryExpr {
                    span,
                    op: UnaryOp::Not,
                    expr: Box::new(comparison),
                })
            } else {
                comparison
            };
            chain = Some(match chain {
                Some(previous) => binary(previous, BinaryOp::And, comparison),
                None => comparison,
            });
            left = right;
        }
        Ok(chain.unwrap_or(left))
    }

    pub(super) fn parse_addition(&mut self) -> Result<Expr, ParseError> {
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

    pub(super) fn parse_multiplication(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_unary()?;
        loop {
            let op = if self.take_simple(&TokenKind::Star).is_some() {
                Some(BinaryOp::Mul)
            } else if matches!(&self.peek().kind, TokenKind::Identifier(name) if name == "X") {
                self.advance();
                Some(BinaryOp::MatMul)
            } else if self.take_simple(&TokenKind::Caret).is_some() {
                Some(BinaryOp::Cross)
            } else if self.take_simple(&TokenKind::Slash).is_some() {
                Some(BinaryOp::Div)
            } else if self.take_simple(&TokenKind::Percent).is_some() {
                Some(BinaryOp::Mod)
            } else {
                None
            };
            let Some(op) = op else { break };
            expression = binary(expression, op, self.parse_unary()?);
        }
        Ok(expression)
    }

    pub(super) fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_postfix()?;
        if self.take_simple(&TokenKind::Power).is_some() {
            return Ok(binary(left, BinaryOp::Power, self.parse_unary()?));
        }
        Ok(left)
    }

    pub(super) fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.at(&TokenKind::Await) {
            let start = self.advance().span.start;
            let first = self.parse_unary()?;
            let mut values = vec![first];
            while self.take_simple(&TokenKind::Comma).is_some() {
                values.push(self.parse_unary()?);
            }
            let mut end = values.last().unwrap().span().end;
            if self.take_simple(&TokenKind::With).is_some() {
                let capability = self.expect_identifier("await capability")?;
                end = capability.span.end;
                while self.take_simple(&TokenKind::And).is_some() {
                    end = self.expect_identifier("await capability")?.span.end;
                }
            }
            let value = if values.len() == 1 {
                values.pop().unwrap()
            } else {
                Expr::Tuple(CollectionExpr {
                    span: Span::new(start, end),
                    elements: values,
                })
            };
            let end = value.span().end;
            return Ok(Expr::Await(AwaitExpr {
                span: Span::new(start, end),
                value: Box::new(value),
            }));
        }
        if self.at(&TokenKind::Async) {
            let start = self.advance().span.start;
            if self.at(&TokenKind::Send) {
                let value = self.parse_send()?;
                let end = value.span().end;
                return Ok(Expr::Async(AsyncExpr {
                    span: Span::new(start, end),
                    value: Box::new(value),
                    owner: TaskOwner::SelfOwned,
                    placement: TaskPlacement::Default,
                    captures: Vec::new(),
                }));
            }
            let value = self.parse_postfix()?;
            if !self.at(&TokenKind::With) {
                let context = self.task_contexts.last().cloned().ok_or_else(|| {
                    self.error("expected `with` after async operation outside a task context")
                })?;
                let end = value.span().end;
                return Ok(Expr::Async(AsyncExpr {
                    span: Span::new(start, end),
                    value: Box::new(value),
                    owner: context.owner,
                    placement: context.placement,
                    captures: context.captures,
                }));
            }
            self.expect_simple(TokenKind::With, "`with` after async operation")?;
            let owner_name = self.expect_identifier("task owner")?;
            let owner = if owner_name.name == "runtime" {
                TaskOwner::Runtime
            } else {
                TaskOwner::SelfOwned
            };
            let mut placement = task_placement(&owner_name.name).unwrap_or(TaskPlacement::Default);
            if owner_name.name == "fuse" {
                return Err(self.error(
                    "kernel fusion is automatic for compatible model operations; remove `fuse`",
                ));
            }
            let mut end = owner_name.span.end;
            let mut captures = Vec::new();
            while self.take_simple(&TokenKind::And).is_some() {
                let capability = self.expect_identifier("captured capability")?;
                end = capability.span.end;
                if let Some(requested) = task_placement(&capability.name) {
                    if placement != TaskPlacement::Default {
                        return Err(self.error("task placement was specified more than once"));
                    }
                    placement = requested;
                } else if capability.name == "fuse" {
                    return Err(self.error(
                        "kernel fusion is automatic for compatible model operations; remove `fuse`",
                    ));
                } else if !matches!(capability.name.as_str(), "self" | "runtime") {
                    captures.push(capability);
                } else {
                    continue;
                }
            }
            return Ok(Expr::Async(AsyncExpr {
                span: Span::new(start, end),
                value: Box::new(value),
                owner,
                placement,
                captures,
            }));
        }
        if self.at(&TokenKind::Send) {
            return self.parse_send();
        }
        let (op, start) = if self.at(&TokenKind::Minus) {
            (Some(UnaryOp::Negate), self.advance().span.start)
        } else if self.at(&TokenKind::Not) {
            (Some(UnaryOp::Not), self.advance().span.start)
        } else {
            (None, 0)
        };
        if let Some(op) = op {
            let expr = self.parse_unary()?;
            let end = expr.span().end;
            Ok(Expr::Unary(UnaryExpr {
                span: Span::new(start, end),
                op,
                expr: Box::new(expr),
            }))
        } else {
            let ownership = if self.at(&TokenKind::View) {
                Some(OwnershipOp::View)
            } else if self.at(&TokenKind::Borrow) {
                Some(OwnershipOp::Borrow)
            } else if self.at(&TokenKind::Clone) {
                Some(OwnershipOp::Clone)
            } else if self.at(&TokenKind::Move) {
                Some(OwnershipOp::Move)
            } else if self.at(&TokenKind::Ampersand) {
                if self.unsafe_depth == 0 {
                    return Err(self.error("address-of is only valid inside `unsafe`"));
                }
                Some(OwnershipOp::AddressOf)
            } else {
                None
            };
            if let Some(op) = ownership {
                let start = self.advance().span.start;
                let value = self.parse_unary()?;
                let end = value.span().end;
                Ok(Expr::Ownership(OwnershipExpr {
                    span: Span::new(start, end),
                    value: Box::new(value),
                    op,
                }))
            } else {
                self.parse_power()
            }
        }
    }

    pub(super) fn parse_send(&mut self) -> Result<Expr, ParseError> {
        let start = self.expect_simple(TokenKind::Send, "`send`")?.span.start;
        let value = self.parse_expression()?;
        self.expect_simple(TokenKind::With, "`with` before send channel")?;
        let channel = self.parse_expression()?;
        let end = channel.span().end;
        Ok(Expr::Send(severian_ast::SendExpr {
            span: Span::new(start, end),
            value: Box::new(value),
            channel: Box::new(channel),
        }))
    }

    pub(super) fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_primary()?;
        loop {
            if self.take_simple(&TokenKind::LeftParen).is_some() {
                let start = expression.span().start;
                let mut args = Vec::new();
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        let name = if matches!(self.peek().kind, TokenKind::Identifier(_))
                            && self.peek_kind(1, &TokenKind::Equal)
                        {
                            let name = self.expect_identifier("argument name")?;
                            self.expect_simple(TokenKind::Equal, "`=`")?;
                            Some(name)
                        } else {
                            None
                        };
                        let value = self.parse_expression()?;
                        args.push(CallArg {
                            span: value.span(),
                            name,
                            value,
                        });
                        if self.take_simple(&TokenKind::Comma).is_none() {
                            break;
                        }
                        if self.at(&TokenKind::RightParen) {
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
            } else if self.take_simple(&TokenKind::LeftBracket).is_some() {
                let start = expression.span().start;
                let first = if self.at(&TokenKind::Colon) {
                    None
                } else {
                    Some(self.parse_expression()?)
                };
                if self.take_simple(&TokenKind::Colon).is_some() {
                    let slice_end =
                        if self.at(&TokenKind::Colon) || self.at(&TokenKind::RightBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expression()?))
                        };
                    let step = if self.take_simple(&TokenKind::Colon).is_some() {
                        if self.at(&TokenKind::RightBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expression()?))
                        }
                    } else {
                        None
                    };
                    let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
                    expression = Expr::Slice(SliceExpr {
                        span: Span::new(start, end),
                        object: Box::new(expression),
                        start: first.map(Box::new),
                        end: slice_end,
                        step,
                    });
                } else {
                    let index = first.ok_or_else(|| ParseError {
                        span: self.peek().span,
                        message: "an index expression cannot be empty".into(),
                    })?;
                    let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
                    expression = Expr::Index(IndexExpr {
                        span: Span::new(start, end),
                        object: Box::new(expression),
                        index: Box::new(index),
                    });
                }
            } else if self.take_simple(&TokenKind::Dot).is_some() {
                let start = expression.span().start;
                let member = self.expect_identifier("member name")?;
                expression = Expr::Member(MemberExpr {
                    span: Span::new(start, member.span.end),
                    object: Box::new(expression),
                    member,
                });
            } else {
                break;
            }
        }
        Ok(expression)
    }

    pub(super) fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::When => {
                if self.test_depth == 0 {
                    return Err(ParseError {
                        span: token.span,
                        message: "chaos injection patterns are only valid inside tests".into(),
                    });
                }
                let function = self.parse_postfix()?;
                let action = if self.take_simple(&TokenKind::Return).is_some() {
                    ChaosAction::Return
                } else if self.take_simple(&TokenKind::Throw).is_some() {
                    ChaosAction::Throw
                } else {
                    return Err(self.error("expected `return` or `throw` in chaos pattern"));
                };
                let value = self.parse_expression()?;
                Ok(Expr::ChaosRule(ChaosRuleExpr {
                    span: Span::new(token.span.start, value.span().end),
                    function: Box::new(function),
                    action,
                    value: Box::new(value),
                }))
            }
            TokenKind::Identifier(name)
                if name == "Channel" && self.at(&TokenKind::LeftBracket) =>
            {
                self.advance();
                let element_type = self.parse_type()?;
                self.expect_simple(TokenKind::RightBracket, "`]` after channel type")?;
                self.expect_simple(TokenKind::With, "`with` after channel type")?;
                let buffer = self.expect_identifier("Buffer")?;
                if buffer.name != "Buffer" {
                    return Err(ParseError {
                        span: buffer.span,
                        message: "expected `Buffer`".into(),
                    });
                }
                self.expect_simple(TokenKind::LeftParen, "`(` after Buffer")?;
                let capacity = self.parse_expression()?;
                let end = self
                    .expect_simple(TokenKind::RightParen, "`)` after buffer capacity")?
                    .span
                    .end;
                Ok(Expr::Channel(severian_ast::ChannelExpr {
                    span: Span::new(token.span.start, end),
                    element_type,
                    capacity: Box::new(capacity),
                }))
            }
            TokenKind::Identifier(name) => Ok(Expr::Identifier(Ident {
                span: token.span,
                name,
            })),
            TokenKind::Integer(value) => Ok(Expr::Literal(Literal::Integer {
                span: token.span,
                value,
            })),
            TokenKind::Quantity(value, unit) => Ok(Expr::Literal(Literal::Integer {
                span: token.span,
                value: normalize_quantity(value, &unit, token.span)?,
            })),
            TokenKind::Float(bits) => Ok(Expr::Literal(Literal::Float {
                span: token.span,
                value: f64::from_bits(bits),
            })),
            TokenKind::String(value) => Ok(Expr::Literal(Literal::String {
                span: token.span,
                value,
            })),
            TokenKind::FormattedString(value) => Ok(internal_call(
                "__format",
                token.span,
                vec![Expr::Literal(Literal::String {
                    span: token.span,
                    value,
                })],
            )),
            TokenKind::True => Ok(Expr::Literal(Literal::Boolean {
                span: token.span,
                value: true,
            })),
            TokenKind::False => Ok(Expr::Literal(Literal::Boolean {
                span: token.span,
                value: false,
            })),
            TokenKind::Pipe => {
                let mut params = Vec::new();
                if !self.at(&TokenKind::Pipe) {
                    loop {
                        let name = self.expect_identifier("lambda parameter")?;
                        params.push(Parameter {
                            span: name.span,
                            name,
                            ty: None,
                            default: None,
                        });
                        if self.take_simple(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect_simple(TokenKind::Pipe, "`|` after lambda parameters")?;
                let body = self.parse_expression()?;
                Ok(Expr::Lambda(severian_ast::LambdaExpr {
                    span: Span::new(token.span.start, body.span().end),
                    params,
                    return_type: None,
                    body: LambdaBody::Expr(Box::new(body)),
                }))
            }
            TokenKind::LeftParen => self.parse_parenthesized(token.span.start),
            TokenKind::LeftBracket => self.parse_list(token.span.start),
            TokenKind::LeftBrace => self.parse_braces(token.span.start),
            _ => Err(ParseError {
                span: token.span,
                message: "expected an expression".into(),
            }),
        }
    }

    pub(super) fn parse_parenthesized(&mut self, start: usize) -> Result<Expr, ParseError> {
        let first = self.parse_expression()?;
        if self.take_simple(&TokenKind::Comma).is_none() {
            self.expect_simple(TokenKind::RightParen, "`)`")?;
            return Ok(first);
        }
        let mut elements = vec![first];
        while !self.at(&TokenKind::RightParen) {
            elements.push(self.parse_expression()?);
            if self.take_simple(&TokenKind::Comma).is_none() {
                break;
            }
        }
        let end = self.expect_simple(TokenKind::RightParen, "`)`")?.span.end;
        Ok(Expr::Tuple(CollectionExpr {
            span: Span::new(start, end),
            elements,
        }))
    }

    pub(super) fn parse_list(&mut self, start: usize) -> Result<Expr, ParseError> {
        if self.at(&TokenKind::RightBracket) {
            let end = self.advance().span.end;
            return Ok(Expr::List(CollectionExpr {
                span: Span::new(start, end),
                elements: Vec::new(),
            }));
        }
        let first = self.parse_expression()?;
        if self.take_simple(&TokenKind::For).is_some() {
            let clauses = self.parse_comprehension_clauses()?;
            let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
            return Ok(Expr::ListComprehension(ListComprehensionExpr {
                span: Span::new(start, end),
                element: Box::new(first),
                clauses,
            }));
        }
        let mut elements = vec![first];
        while self.take_simple(&TokenKind::Comma).is_some() {
            if self.at(&TokenKind::RightBracket) {
                break;
            }
            elements.push(self.parse_expression()?);
        }
        let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
        Ok(Expr::List(CollectionExpr {
            span: Span::new(start, end),
            elements,
        }))
    }

    pub(super) fn parse_braces(&mut self, start: usize) -> Result<Expr, ParseError> {
        if self.at(&TokenKind::RightBrace) {
            let end = self.advance().span.end;
            return Ok(Expr::Map(MapExpr {
                span: Span::new(start, end),
                entries: Vec::new(),
            }));
        }
        let first = self.parse_expression()?;
        if self.take_simple(&TokenKind::Colon).is_some() {
            let value = self.parse_expression()?;
            if self.take_simple(&TokenKind::For).is_some() {
                let clauses = self.parse_comprehension_clauses()?;
                let end = self.expect_simple(TokenKind::RightBrace, "`}`")?.span.end;
                return Ok(Expr::MapComprehension(MapComprehensionExpr {
                    span: Span::new(start, end),
                    key: Box::new(first),
                    value: Box::new(value),
                    clauses,
                }));
            }
            let mut entries = vec![MapEntry {
                span: Span::new(first.span().start, value.span().end),
                key: first,
                value,
            }];
            while self.take_simple(&TokenKind::Comma).is_some() {
                if self.at(&TokenKind::RightBrace) {
                    break;
                }
                let key = self.parse_expression()?;
                self.expect_simple(TokenKind::Colon, "`:` between map key and value")?;
                let value = self.parse_expression()?;
                entries.push(MapEntry {
                    span: Span::new(key.span().start, value.span().end),
                    key,
                    value,
                });
            }
            let end = self.expect_simple(TokenKind::RightBrace, "`}`")?.span.end;
            Ok(Expr::Map(MapExpr {
                span: Span::new(start, end),
                entries,
            }))
        } else {
            if self.take_simple(&TokenKind::For).is_some() {
                let clauses = self.parse_comprehension_clauses()?;
                let end = self.expect_simple(TokenKind::RightBrace, "`}`")?.span.end;
                return Ok(Expr::SetComprehension(SetComprehensionExpr {
                    span: Span::new(start, end),
                    element: Box::new(first),
                    clauses,
                }));
            }
            let mut elements = vec![first];
            while self.take_simple(&TokenKind::Comma).is_some() {
                if self.at(&TokenKind::RightBrace) {
                    break;
                }
                elements.push(self.parse_expression()?);
            }
            let end = self.expect_simple(TokenKind::RightBrace, "`}`")?.span.end;
            Ok(Expr::Set(CollectionExpr {
                span: Span::new(start, end),
                elements,
            }))
        }
    }

    pub(super) fn parse_comprehension_clauses(
        &mut self,
    ) -> Result<Vec<ComprehensionClause>, ParseError> {
        let mut clauses = Vec::new();
        loop {
            let first = self.expect_identifier("comprehension variable")?;
            let mut identifiers = vec![first];
            while self.take_simple(&TokenKind::Comma).is_some() {
                identifiers.push(self.expect_identifier("comprehension variable")?);
            }
            let pattern = if identifiers.len() == 1 {
                let identifier = identifiers.pop().unwrap();
                if identifier.name == "_" {
                    Pattern::Wildcard(identifier.span)
                } else {
                    Pattern::Identifier(identifier)
                }
            } else {
                Pattern::Tuple {
                    span: Span::new(
                        identifiers.first().unwrap().span.start,
                        identifiers.last().unwrap().span.end,
                    ),
                    elements: identifiers
                        .into_iter()
                        .map(|identifier| {
                            if identifier.name == "_" {
                                Pattern::Wildcard(identifier.span)
                            } else {
                                Pattern::Identifier(identifier)
                            }
                        })
                        .collect(),
                }
            };
            self.expect_simple(TokenKind::In, "`in`")?;
            let iterable = self.parse_or()?;
            let condition = if self.take_simple(&TokenKind::If).is_some() {
                Some(Box::new(self.parse_or()?))
            } else {
                None
            };
            clauses.push(ComprehensionClause {
                pattern,
                iterable: Box::new(iterable),
                condition,
            });
            if self.take_simple(&TokenKind::For).is_none() {
                break;
            }
        }
        Ok(clauses)
    }
}
