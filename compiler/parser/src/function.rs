use super::*;

impl Parser<'_> {
    pub(super) fn parse_typed_global(&mut self) -> Result<Stmt, ParseError> {
        let ty = self.parse_type()?;
        let name = self.expect_identifier("constant name")?;
        let start = ty.span().start;
        self.expect_simple(TokenKind::Equal, "`=`")?;
        let value = self.parse_expression()?;
        let end = self
            .expect_simple(TokenKind::Newline, "newline after constant")?
            .span
            .end;
        Ok(Stmt::Let(LetStmt {
            span: Span::new(start, end),
            kind: LetKind::Stable,
            name,
            ty: Some(ty),
            value: Some(value),
        }))
    }

    pub(super) fn parse_function(&mut self) -> Result<FunctionDecl, ParseError> {
        self.parse_function_with_decorators(Vec::new())
    }

    pub(super) fn parse_unsafe_extern_block(&mut self) -> Result<Vec<Item>, ParseError> {
        self.expect_simple(TokenKind::Unsafe, "`unsafe`")?;
        self.expect_simple(TokenKind::Colon, "`:` after unsafe")?;
        self.expect_simple(TokenKind::Newline, "newline after unsafe header")?;
        self.expect_simple(TokenKind::Indent, "indented unsafe body")?;

        let mut functions = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if !self.at(&TokenKind::Extern) {
                return Err(self
                    .error("module-level `unsafe:` blocks may only contain extern declarations"));
            }
            let start = self.peek().span.start;
            functions.push(self.parse_extern_function(start)?);
        }
        self.expect_simple(TokenKind::Dedent, "end of unsafe body")?;

        if functions.is_empty() {
            return Err(self.error("module-level `unsafe:` blocks require an extern declaration"));
        }
        while self.at(&TokenKind::Test) {
            functions
                .last_mut()
                .expect("an unsafe extern block has at least one declaration")
                .tests
                .push(self.parse_test()?);
        }

        Ok(functions.into_iter().map(Item::Function).collect())
    }

    pub(super) fn parse_extern_function(
        &mut self,
        start: usize,
    ) -> Result<FunctionDecl, ParseError> {
        self.expect_simple(TokenKind::Extern, "`extern` inside `unsafe:`")?;
        self.expect_simple(TokenKind::LeftParen, "`(` after `extern`")?;
        let symbol = match self.advance().clone() {
            Token {
                kind: TokenKind::String(symbol),
                ..
            } => symbol,
            _ => return Err(self.error("extern declarations require a linker symbol string")),
        };
        self.expect_simple(TokenKind::RightParen, "`)` after extern linker symbol")?;
        self.expect_simple(TokenKind::Def, "`def` after extern linker symbol")?;
        let name = self.expect_identifier("extern function name")?;
        let generic_params = self.parse_generic_parameters()?;
        let params = self.parse_parameters()?;
        if let Some(parameter) = params.iter().find(|parameter| parameter.ty.is_none()) {
            return Err(ParseError {
                span: parameter.name.span,
                message: "extern ABI parameters require explicit types".into(),
            });
        }
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let end = self
            .expect_simple(TokenKind::Newline, "newline after extern declaration")?
            .span
            .end;
        let mut tests = Vec::new();
        while self.at(&TokenKind::Test) {
            tests.push(self.parse_test()?);
        }
        Ok(FunctionDecl {
            span: Span::new(start, end),
            native_symbol: Some(symbol),
            decorators: Vec::new(),
            name,
            generic_params,
            params,
            return_type,
            contract: None,
            body: Block {
                span: Span::new(end, end),
                statements: Vec::new(),
            },
            tests,
        })
    }

    pub(super) fn parse_generic_parameters(&mut self) -> Result<Vec<GenericParameter>, ParseError> {
        if self.take_simple(&TokenKind::LeftBracket).is_none() {
            return Ok(Vec::new());
        }
        let mut parameters = Vec::new();
        while !self.at(&TokenKind::RightBracket) {
            let name = self.expect_identifier("generic parameter")?;
            let start = name.span.start;
            let mut constraints = Vec::new();
            if self.take_simple(&TokenKind::Colon).is_some() {
                constraints.push(self.parse_type()?);
                while self.take_simple(&TokenKind::Plus).is_some() {
                    constraints.push(self.parse_type()?);
                }
            }
            let end = constraints
                .last()
                .map_or(name.span.end, |constraint| constraint.span().end);
            parameters.push(GenericParameter {
                span: Span::new(start, end),
                name,
                constraints,
            });
            if self.take_simple(&TokenKind::Comma).is_none() {
                break;
            }
        }
        self.expect_simple(TokenKind::RightBracket, "`]`")?;
        Ok(parameters)
    }

    pub(super) fn parse_function_with_decorators(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<FunctionDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Def, "`def`")?.span.start;
        let name = self.expect_identifier("function name")?;
        let generic_params = self.parse_generic_parameters()?;
        let params = self.parse_parameters()?;
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let contract = self.parse_optional_contract()?;
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
            native_symbol: None,
            decorators,
            name,
            generic_params,
            params,
            return_type,
            contract,
            body,
            tests,
        })
    }

    pub(super) fn parse_optional_contract(
        &mut self,
    ) -> Result<Option<FunctionContract>, ParseError> {
        if self.take_simple(&TokenKind::With).is_some() {
            self.take_simple(&TokenKind::Newline);
            return self.parse_function_contract().map(Some);
        }
        if self.at(&TokenKind::Newline) && self.peek_kind(1, &TokenKind::LeftBrace) {
            return Err(self.error("API contracts require `with` before `{`"));
        }
        if self.at(&TokenKind::LeftBrace) {
            return Err(self.error("API contracts require `with` before `{`"));
        }
        Ok(None)
    }

    pub(super) fn parse_function_contract(&mut self) -> Result<FunctionContract, ParseError> {
        let start = self
            .expect_simple(TokenKind::LeftBrace, "`{` at the start of the contract")?
            .span
            .start;
        self.skip_parenthesized_layout();
        let mut clauses = Vec::new();
        let mut capabilities = Vec::new();
        while !self.at(&TokenKind::RightBrace) {
            if self.take_simple(&TokenKind::With).is_some() {
                capabilities.push(self.expect_identifier("function capability")?);
            } else {
                let clause_start = self.peek().span.start;
                let deferred = self.take_simple(&TokenKind::Defer).is_some();
                let condition = self.parse_expression()?;
                let failure = if self.take_simple(&TokenKind::Arrow).is_some() {
                    Some(self.parse_contract_failure()?)
                } else {
                    None
                };
                let clause_end = failure
                    .as_ref()
                    .map_or(condition.span().end, |failure| failure.span.end);
                clauses.push(ContractClause {
                    span: Span::new(clause_start, clause_end),
                    deferred,
                    condition,
                    failure,
                });
            }
            if self.take_simple(&TokenKind::Comma).is_some() {
                self.skip_parenthesized_layout();
            } else if !self.at(&TokenKind::RightBrace) {
                return Err(self.error("expected `,` between contract clauses"));
            }
        }
        let end = self
            .expect_simple(TokenKind::RightBrace, "`}` after function contract")?
            .span
            .end;
        Ok(FunctionContract {
            span: Span::new(start, end),
            clauses,
            capabilities,
        })
    }

    pub(super) fn parse_contract_failure(&mut self) -> Result<ContractFailure, ParseError> {
        let exception = self.expect_identifier("`exception` after `->`")?;
        if exception.name != "exception" {
            return Err(ParseError {
                span: exception.span,
                message: "a contract failure action must be `exception(...)`".into(),
            });
        }
        self.expect_simple(TokenKind::LeftParen, "`(` after `exception`")?;
        let message_token = self.advance().clone();
        let message = match message_token.kind {
            TokenKind::String(message) => message,
            _ => {
                return Err(ParseError {
                    span: message_token.span,
                    message: "a contract exception requires a string message".into(),
                })
            }
        };
        let mut location = false;
        let mut vars = false;
        while self.take_simple(&TokenKind::Comma).is_some() {
            let option = self.expect_identifier("`location` or `vars`")?;
            match option.name.as_str() {
                "location" if !location => location = true,
                "vars" if !vars => vars = true,
                "location" | "vars" => {
                    return Err(ParseError {
                        span: option.span,
                        message: format!("duplicate `{}` contract exception option", option.name),
                    })
                }
                _ => {
                    return Err(ParseError {
                        span: option.span,
                        message: "contract exception options are `location` and `vars`".into(),
                    })
                }
            }
        }
        let end = self
            .expect_simple(TokenKind::RightParen, "`)` after contract exception")?
            .span
            .end;
        Ok(ContractFailure {
            span: Span::new(exception.span.start, end),
            message,
            location,
            vars,
        })
    }

    pub(super) fn parse_decorators(&mut self) -> Result<Vec<Decorator>, ParseError> {
        let mut decorators = Vec::new();
        while self.at(&TokenKind::At) {
            let start = self.expect_simple(TokenKind::At, "`@`")?.span.start;
            let segments = self.parse_path()?;
            let name_start = segments.first().unwrap().span.start;
            let name_end = segments.last().unwrap().span.end;
            let mut symbols = Vec::new();
            let end = if self.take_simple(&TokenKind::LeftParen).is_some() {
                if self.at(&TokenKind::RightParen) {
                    return Err(self.error(
                        "empty decorator arguments are not allowed; write the decorator without `()`",
                    ));
                }
                loop {
                    let token = self.advance().clone();
                    let spelling = match token.kind {
                        TokenKind::Identifier(value) => value,
                        TokenKind::Star => "*".into(),
                        TokenKind::Caret => "^".into(),
                        TokenKind::Pipe => "|".into(),
                        TokenKind::Ampersand => "&".into(),
                        TokenKind::Plus => "+".into(),
                        TokenKind::Minus => "-".into(),
                        TokenKind::Slash => "/".into(),
                        TokenKind::Percent => "%".into(),
                        _ => {
                            return Err(ParseError {
                                span: token.span,
                                message: "expected decorator symbol".into(),
                            })
                        }
                    };
                    symbols.push(DecoratorSymbol {
                        span: token.span,
                        spelling,
                    });
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
                self.expect_simple(TokenKind::RightParen, "`)` after decorator symbols")?
                    .span
                    .end
            } else {
                name_end
            };
            self.expect_simple(TokenKind::Newline, "newline after decorator")?;
            decorators.push(Decorator {
                span: Span::new(start, end),
                name: TypePath {
                    span: Span::new(name_start, name_end),
                    segments,
                    args: Vec::new(),
                },
                symbols,
            });
        }
        Ok(decorators)
    }

    pub(super) fn parse_parameters(&mut self) -> Result<Vec<Parameter>, ParseError> {
        self.expect_simple(TokenKind::LeftParen, "`(`")?;
        self.skip_parenthesized_layout();
        let mut params = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let param_start = self.peek().span.start;
                let param_name = self.expect_identifier("parameter name")?;
                let ty = if self.take_simple(&TokenKind::Colon).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let default = if self.take_simple(&TokenKind::Equal).is_some() {
                    Some(self.parse_expression()?)
                } else {
                    None
                };
                let end = default.as_ref().map_or_else(
                    || ty.as_ref().map_or(param_name.span.end, |ty| ty.span().end),
                    |value| value.span().end,
                );
                params.push(Parameter {
                    span: Span::new(param_start, end),
                    name: param_name,
                    ty,
                    default,
                });
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
                self.skip_parenthesized_layout();
                if self.at(&TokenKind::RightParen) {
                    break;
                }
            }
        }
        self.skip_parenthesized_layout();
        self.expect_simple(TokenKind::RightParen, "`)`")?;
        Ok(params)
    }

    pub(super) fn skip_parenthesized_layout(&mut self) {
        while matches!(
            self.peek().kind,
            TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent
        ) {
            self.advance();
        }
    }

    pub(super) fn parse_test(&mut self) -> Result<TestBlock, ParseError> {
        let start = self.expect_simple(TokenKind::Test, "`test`")?.span.start;
        let mut modes = Vec::new();
        if self.at(&TokenKind::With) && self.starts_test_modes() {
            self.advance();
            loop {
                let mode = self.expect_identifier("test mode")?;
                modes.push(match mode.name.as_str() {
                    "property" => TestMode::Property,
                    "bench" => TestMode::Bench,
                    "chaos" => TestMode::Chaos,
                    "integration" => TestMode::Integration,
                    "profile" => TestMode::Profile,
                    _ => {
                        return Err(ParseError {
                            span: mode.span,
                            message: format!("unknown test mode `{}`", mode.name),
                        })
                    }
                });
                if self.take_simple(&TokenKind::And).is_none() {
                    break;
                }
            }
        }
        let name = if let TokenKind::String(value) = self.peek().kind.clone() {
            let token = self.advance().clone();
            Some(Ident {
                span: token.span,
                name: value,
            })
        } else {
            None
        };
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let contract = self.parse_optional_contract()?;
        self.expect_simple(TokenKind::Colon, "`:` after test")?;
        self.expect_simple(TokenKind::Newline, "newline after test header")?;
        self.expect_simple(TokenKind::Indent, "indented test body")?;
        self.test_depth += 1;
        let body = self.parse_block();
        self.test_depth -= 1;
        let body = body?;
        let end = self
            .expect_simple(TokenKind::Dedent, "end of test body")?
            .span
            .end;
        Ok(TestBlock {
            span: Span::new(start, end),
            modes,
            name,
            return_type,
            contract,
            body,
        })
    }

    pub(super) fn starts_test_modes(&self) -> bool {
        matches!(
            &self.peek_token(1).kind,
            TokenKind::Identifier(name)
                if matches!(
                    name.as_str(),
                    "property" | "bench" | "chaos" | "integration" | "profile"
                )
        )
    }
}
