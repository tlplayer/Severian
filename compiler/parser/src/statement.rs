use super::*;
use crate::cursor::{is_task_context_symbol, task_placement};

impl Parser<'_> {
    pub(super) fn parse_block(&mut self) -> Result<Block, ParseError> {
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

    pub(super) fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        if self.at(&TokenKind::Def) {
            return self.parse_function().map(Stmt::Function);
        }
        if self.at(&TokenKind::Break) || self.at(&TokenKind::Continue) {
            let token = self.advance().clone();
            if self.loop_depth == 0 {
                return Err(ParseError {
                    span: token.span,
                    message: "loop control is only valid inside a loop".into(),
                });
            }
            let end = self
                .expect_simple(TokenKind::Newline, "newline after loop control")?
                .span
                .end;
            let span = Span::new(token.span.start, end);
            return Ok(if token.kind == TokenKind::Break {
                Stmt::Break(span)
            } else {
                Stmt::Continue(span)
            });
        }
        if self.at(&TokenKind::If) {
            return self.parse_if().map(Stmt::If);
        }
        if self.at(&TokenKind::While) {
            return self.parse_while().map(Stmt::While);
        }
        if self.at(&TokenKind::For) {
            return self.parse_for();
        }
        if self.at(&TokenKind::Switch)
            || (matches!(&self.peek().kind, TokenKind::Identifier(name) if name == "match")
                && !self.peek_kind(1, &TokenKind::LeftParen))
        {
            return self.parse_switch().map(Stmt::Switch);
        }
        if self.at(&TokenKind::With) {
            let start = self.advance().span.start;
            let mut resources = vec![Expr::Identifier(self.expect_identifier("with resource")?)];
            while self.take_simple(&TokenKind::And).is_some() {
                resources.push(Expr::Identifier(self.expect_identifier("with resource")?));
            }
            self.expect_simple(TokenKind::Colon, "`:` after with resources")?;
            let identifiers = resources
                .iter()
                .filter_map(|resource| {
                    let Expr::Identifier(identifier) = resource else {
                        return None;
                    };
                    Some(identifier.clone())
                })
                .collect::<Vec<_>>();
            if identifiers
                .iter()
                .any(|identifier| identifier.name == "fuse")
            {
                return Err(self.error(
                    "kernel fusion is automatic for compatible model operations; remove `fuse`",
                ));
            }
            let owner = if identifiers
                .iter()
                .any(|identifier| identifier.name == "runtime")
            {
                Some(TaskOwner::Runtime)
            } else if identifiers
                .iter()
                .any(|identifier| identifier.name == "self")
            {
                Some(TaskOwner::SelfOwned)
            } else {
                None
            };
            let placements = identifiers
                .iter()
                .filter_map(|identifier| task_placement(&identifier.name))
                .collect::<Vec<_>>();
            if placements.len() > 1 {
                return Err(self.error("a task context accepts only one execution placement"));
            }
            let placement = placements.first().copied();
            let establishes_task_context = owner.is_some() || placement.is_some();
            if establishes_task_context {
                let captures = identifiers
                    .iter()
                    .filter(|identifier| !is_task_context_symbol(&identifier.name))
                    .cloned()
                    .collect();
                self.task_contexts.push(TaskContext {
                    owner: owner.unwrap_or(TaskOwner::SelfOwned),
                    placement: placement.unwrap_or(TaskPlacement::Default),
                    captures,
                });
            }
            let body = self.parse_suite("with");
            if establishes_task_context {
                self.task_contexts.pop();
            }
            let body = body?;
            return Ok(Stmt::With(WithBlock {
                span: Span::new(start, body.span.end),
                resources,
                body,
            }));
        }
        if self.at(&TokenKind::Unsafe) {
            if self.test_depth > 0 {
                return Err(self.error("tests may not contain `unsafe` blocks"));
            }
            let start = self.advance().span.start;
            self.expect_simple(TokenKind::Colon, "`:` after unsafe")?;
            self.unsafe_depth += 1;
            let body = self.parse_suite("unsafe");
            self.unsafe_depth -= 1;
            let body = body?;
            return Ok(Stmt::Unsafe(UnsafeBlock {
                span: Span::new(start, body.span.end),
                body,
            }));
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
            let parenthesized = self.take_simple(&TokenKind::LeftParen).is_some();
            let condition = self.parse_expression()?;
            let message = if self.take_simple(&TokenKind::Comma).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            if parenthesized {
                self.expect_simple(TokenKind::RightParen, "`)` after assertion")?;
            }
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
            && self.peek_kind(1, &TokenKind::TryEqual)
        {
            let name = self.expect_identifier("try binding name")?;
            let start = name.span.start;
            self.expect_simple(TokenKind::TryEqual, "`?=`")?;
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after try binding")?
                .span
                .end;
            return Ok(Stmt::TryBind(severian_ast::TryBindStmt {
                span: Span::new(start, end),
                name,
                ty: None,
                value,
            }));
        }
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.peek_kind(1, &TokenKind::Colon)
        {
            let name = self.expect_identifier("binding name")?;
            let start = name.span.start;
            self.expect_simple(TokenKind::Colon, "`:` after binding name")?;
            let ty = self.parse_type()?;
            if self.at(&TokenKind::Newline) {
                return Err(ParseError {
                    span: name.span,
                    message: format!("E000205: binding `{}` requires an initializer", name.name),
                });
            }
            self.expect_simple(TokenKind::Equal, "`=` after binding type")?;
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after typed binding")?
                .span
                .end;
            return Ok(Stmt::Let(LetStmt {
                span: Span::new(start, end),
                kind: LetKind::Stable,
                name,
                ty: Some(ty),
                value: Some(value),
            }));
        }
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.peek_kind(1, &TokenKind::Comma)
        {
            let mut names = vec![self.expect_identifier("destructured binding")?];
            while self.take_simple(&TokenKind::Comma).is_some() {
                names.push(self.expect_identifier("destructured binding")?);
            }
            let start = names.first().unwrap().span.start;
            self.expect_simple(TokenKind::Equal, "`=` after destructured bindings")?;
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after destructured binding")?
                .span
                .end;
            return Ok(Stmt::DestructureLet(DestructureLetStmt {
                span: Span::new(start, end),
                names,
                value,
            }));
        }
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && (self.peek_kind(1, &TokenKind::Equal)
                || self.peek_kind(1, &TokenKind::ChangeableEqual))
        {
            let name = self.expect_identifier("binding name")?;
            let start = name.span.start;
            let kind = if self.take_simple(&TokenKind::ChangeableEqual).is_some() {
                LetKind::Changeable
            } else {
                self.expect_simple(TokenKind::Equal, "`=`")?;
                LetKind::Stable
            };
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after binding")?
                .span
                .end;
            return Ok(Stmt::Let(LetStmt {
                span: Span::new(start, end),
                kind,
                name,
                ty: None,
                value: Some(value),
            }));
        }

        let target = self.parse_expression()?;
        let op = self.take_assign_op();
        if let Some(op) = op {
            let value = self.parse_expression()?;
            let end = self
                .expect_simple(TokenKind::Newline, "newline after assignment")?
                .span
                .end;
            return Ok(Stmt::Assign(AssignStmt {
                span: Span::new(target.span().start, end),
                target,
                op,
                value,
            }));
        }
        self.expect_simple(TokenKind::Newline, "newline after statement")?;
        Ok(Stmt::Expr(target))
    }

    pub(super) fn parse_if(&mut self) -> Result<IfStmt, ParseError> {
        let start = self.expect_simple(TokenKind::If, "`if`")?.span.start;
        self.parse_conditional_branch(start, "if")
    }

    pub(super) fn parse_conditional_branch(
        &mut self,
        start: usize,
        spelling: &str,
    ) -> Result<IfStmt, ParseError> {
        let condition = self.parse_expression()?;
        self.expect_simple(TokenKind::Colon, "`:` after conditional condition")?;
        let then_block = self.parse_suite(spelling)?;
        let mut end = then_block.span.end;
        let else_branch = if let Some(start) = self
            .take_simple(&TokenKind::Elif)
            .map(|token| token.span.start)
        {
            let branch = self.parse_conditional_branch(start, "elif")?;
            end = branch.span.end;
            Some(ElseBranch::If(Box::new(branch)))
        } else if let Some(start) = self
            .take_simple(&TokenKind::Else)
            .map(|token| token.span.start)
        {
            if self.at(&TokenKind::If) {
                let branch = self.parse_if()?;
                end = branch.span.end;
                Some(ElseBranch::If(Box::new(branch)))
            } else if !self.at(&TokenKind::Colon) {
                let branch = self.parse_conditional_branch(start, "else")?;
                end = branch.span.end;
                Some(ElseBranch::If(Box::new(branch)))
            } else {
                self.expect_simple(TokenKind::Colon, "`:` after else")?;
                let block = self.parse_suite("else")?;
                end = block.span.end;
                Some(ElseBranch::Block(block))
            }
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

    pub(super) fn parse_while(&mut self) -> Result<WhileStmt, ParseError> {
        let start = self.expect_simple(TokenKind::While, "`while`")?.span.start;
        let condition = self.parse_expression()?;
        let mut capabilities = Vec::new();
        let setup = if self.take_simple(&TokenKind::With).is_some() {
            let name = self.expect_identifier("while setup binding")?;
            let setup_start = name.span.start;
            if self.take_simple(&TokenKind::ChangeableEqual).is_some() {
                let value = self.parse_expression()?;
                Some(Box::new(Stmt::Let(LetStmt {
                    span: Span::new(setup_start, value.span().end),
                    kind: LetKind::Changeable,
                    name,
                    ty: None,
                    value: Some(value),
                })))
            } else {
                capabilities.push(Expr::Identifier(name));
                while self.take_simple(&TokenKind::And).is_some() {
                    capabilities.push(Expr::Identifier(
                        self.expect_identifier("while capability")?,
                    ));
                }
                None
            }
        } else {
            None
        };
        self.expect_simple(TokenKind::Colon, "`:` after while")?;
        self.loop_depth += 1;
        let body = self.parse_suite("while");
        self.loop_depth -= 1;
        let body = body?;
        Ok(WhileStmt {
            span: Span::new(start, body.span.end),
            setup,
            capabilities,
            condition,
            body,
        })
    }

    pub(super) fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect_simple(TokenKind::For, "`for`")?.span.start;
        let first = self.expect_identifier("loop variable")?;
        let pattern = if self.take_simple(&TokenKind::Comma).is_some() {
            let second = self.expect_identifier("loop variable")?;
            Pattern::Tuple {
                span: Span::new(first.span.start, second.span.end),
                elements: vec![
                    if first.name == "_" {
                        Pattern::Wildcard(first.span)
                    } else {
                        Pattern::Identifier(first)
                    },
                    if second.name == "_" {
                        Pattern::Wildcard(second.span)
                    } else {
                        Pattern::Identifier(second)
                    },
                ],
            }
        } else if first.name == "_" {
            Pattern::Wildcard(first.span)
        } else {
            Pattern::Identifier(first)
        };
        self.expect_simple(TokenKind::In, "`in`")?;
        let iterable = self.parse_expression()?;
        let mut setup = None;
        let placement = if self.take_simple(&TokenKind::With).is_some() {
            let name = self.expect_identifier("for setup binding or execution placement")?;
            if self.take_simple(&TokenKind::ChangeableEqual).is_some() {
                let value = self.parse_expression()?;
                setup = Some(Box::new(Stmt::Let(LetStmt {
                    span: Span::new(name.span.start, value.span().end),
                    kind: LetKind::Changeable,
                    name,
                    ty: None,
                    value: Some(value),
                })));
                None
            } else if matches!(name.name.as_str(), "gpu" | "simd") {
                Some(name)
            } else {
                return Err(ParseError {
                    span: name.span,
                    message: "a for loop `with` clause requires `name := value`, `gpu`, or `simd`"
                        .into(),
                });
            }
        } else {
            None
        };
        self.expect_simple(TokenKind::Colon, "`:` after for")?;
        self.loop_depth += 1;
        let body = self.parse_suite("for");
        self.loop_depth -= 1;
        let body = body?;
        let end = body.span.end;
        let loop_statement = Stmt::For(ForStmt {
            span: Span::new(start, body.span.end),
            setup,
            pattern,
            iterable,
            body,
        });
        Ok(if let Some(placement) = placement {
            Stmt::With(WithBlock {
                span: Span::new(start, end),
                resources: vec![Expr::Identifier(placement)],
                body: Block {
                    span: Span::new(start, end),
                    statements: vec![loop_statement],
                },
            })
        } else {
            loop_statement
        })
    }

    pub(super) fn parse_switch(&mut self) -> Result<SwitchStmt, ParseError> {
        let python_match =
            matches!(&self.peek().kind, TokenKind::Identifier(name) if name == "match");
        let start = if python_match {
            self.expect_identifier("`match`")?.span
        } else {
            self.expect_simple(TokenKind::Switch, "`switch`")?.span
        }
        .start;
        let mut values = vec![self.parse_equality()?];
        while self.take_simple(&TokenKind::And).is_some() {
            values.push(self.parse_equality()?);
        }
        let repeat_condition = if self.take_simple(&TokenKind::While).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };
        let setup = if self.take_simple(&TokenKind::With).is_some() {
            let name = self.expect_identifier("switch setup binding")?;
            let setup_start = name.span.start;
            self.expect_simple(TokenKind::ChangeableEqual, "`:=` in switch setup")?;
            let value = self.parse_expression()?;
            Some(Box::new(Stmt::Let(LetStmt {
                span: Span::new(setup_start, value.span().end),
                kind: LetKind::Changeable,
                name,
                ty: None,
                value: Some(value),
            })))
        } else {
            None
        };
        self.expect_simple(TokenKind::Colon, "`:` after switch value")?;
        self.expect_simple(TokenKind::Newline, "newline after switch header")?;
        self.expect_simple(TokenKind::Indent, "indented switch arms")?;
        let mut arms = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let arm_start = self.peek().span.start;
            if python_match {
                let case = self.expect_identifier("`case` before match pattern")?;
                if case.name != "case" {
                    return Err(ParseError {
                        span: case.span,
                        message: "expected `case` before match pattern".into(),
                    });
                }
            }
            let mut patterns = vec![self.parse_pattern()?];
            while self.take_simple(&TokenKind::Pipe).is_some() {
                patterns.push(self.parse_pattern()?);
            }
            let source = if self.take_simple(&TokenKind::From).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            let guard = if self.take_simple(&TokenKind::If).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            self.expect_simple(TokenKind::Colon, "`:` after match pattern")?;
            let body = self.parse_suite(if python_match {
                "match case"
            } else {
                "switch arm"
            })?;
            for pattern in patterns {
                arms.push(SwitchArm {
                    span: Span::new(arm_start, body.span.end),
                    source: source.clone(),
                    pattern,
                    guard: guard.clone(),
                    body: body.clone(),
                });
            }
        }
        let end = self
            .expect_simple(TokenKind::Dedent, "end of switch")?
            .span
            .end;
        Ok(SwitchStmt {
            span: Span::new(start, end),
            values,
            repeat_condition,
            setup,
            arms,
        })
    }

    pub(super) fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let token = self.advance().clone();
        match token.kind {
            TokenKind::Identifier(name) if name == "_" => Ok(Pattern::Wildcard(token.span)),
            TokenKind::Identifier(name) => {
                let ident = Ident {
                    span: token.span,
                    name,
                };
                let mut fields = Vec::new();
                let mut end = ident.span.end;
                if self.take_simple(&TokenKind::LeftParen).is_some() {
                    if !self.at(&TokenKind::RightParen) {
                        loop {
                            fields.push(self.parse_pattern()?);
                            if self.take_simple(&TokenKind::Comma).is_none() {
                                break;
                            }
                        }
                    }
                    end = self
                        .expect_simple(TokenKind::RightParen, "`)` in pattern")?
                        .span
                        .end;
                } else if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    fields.push(self.parse_pattern()?);
                    end = fields.last().unwrap().span().end;
                }
                Ok(Pattern::Constructor {
                    span: Span::new(ident.span.start, end),
                    name: Type::Named(TypePath {
                        span: ident.span,
                        segments: vec![ident],
                        args: Vec::new(),
                    }),
                    fields,
                })
            }
            TokenKind::Integer(value) => Ok(Pattern::Literal(Literal::Integer {
                span: token.span,
                value,
            })),
            TokenKind::Quantity(_, _) => Err(ParseError {
                span: token.span,
                message: "profile quantities cannot be used as match patterns".into(),
            }),
            TokenKind::Float(bits) => Ok(Pattern::Literal(Literal::Float {
                span: token.span,
                value: f64::from_bits(bits),
            })),
            TokenKind::String(value) => Ok(Pattern::Literal(Literal::String {
                span: token.span,
                value,
            })),
            TokenKind::True => Ok(Pattern::Literal(Literal::Boolean {
                span: token.span,
                value: true,
            })),
            TokenKind::False => Ok(Pattern::Literal(Literal::Boolean {
                span: token.span,
                value: false,
            })),
            _ => Err(ParseError {
                span: token.span,
                message: "expected a switch pattern".into(),
            }),
        }
    }

    pub(super) fn parse_suite(&mut self, owner: &str) -> Result<Block, ParseError> {
        self.expect_simple(TokenKind::Newline, &format!("newline after {owner} header"))?;
        self.expect_simple(TokenKind::Indent, &format!("indented {owner} body"))?;
        let body = self.parse_block()?;
        self.expect_simple(TokenKind::Dedent, &format!("end of {owner} body"))?;
        Ok(body)
    }
}
