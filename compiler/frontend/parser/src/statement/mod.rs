use severian_ast::{
    BinaryOperator, Binding, CompilerExpectation, CompilerTestCase, Decorator, DecoratorArgument,
    DecoratorValue, Expression, ExpressionKind, FunctionDeclaration, FunctionParameter,
    ImportDeclaration, ImportSubject, Item, Literal, MatchCase, Module, OperatorDeclaration,
    OperatorParameter, OperatorSyntax, PropertyDeclaration, Statement, TestDeclaration,
    TraitDeclaration, TypeAnnotation, TypeAnnotationKind, TypeDeclaration, UnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_lexer::{scan, Token, TokenKind};
use severian_source::{SourceFile, Span};

pub fn parse(tokens: &[Token]) -> Result<Module, Diagnostic> {
    Parser { tokens, cursor: 0 }.module()
}

struct Parser<'a> {
    tokens: &'a [Token],
    cursor: usize,
}

impl Parser<'_> {
    fn module(mut self) -> Result<Module, Diagnostic> {
        let mut module = Module::default();
        self.separators();
        while !self.at(&TokenKind::Eof) {
            let decorators = if self.at(&TokenKind::At) {
                self.decorators()?
            } else {
                Vec::new()
            };
            if self.at_identifier("trait") {
                module
                    .items
                    .push(Item::Trait(self.trait_declaration(decorators)?));
                self.separators();
                continue;
            } else if self.at_identifier("def") {
                let declaration = self.function_declaration(decorators)?;
                let has_body = declaration.body.is_some();
                module.items.push(Item::Function(declaration));
                if has_body {
                    self.separators();
                    continue;
                }
            } else if self.at_identifier("type") {
                module
                    .items
                    .push(Item::Type(self.type_declaration(decorators)?));
            } else if self.at_identifier("test") {
                if !decorators.is_empty() {
                    return Err(self.error("decorators may not precede a test declaration"));
                }
                module.items.push(Item::Test(self.test_declaration()?));
                self.separators();
                continue;
            } else if self.at_identifier("import") {
                if !decorators.is_empty() {
                    return Err(self.error("decorators may only precede declarations"));
                }
                module.items.push(Item::Import(self.import_declaration()?));
            } else {
                if !decorators.is_empty() {
                    return Err(self.error("expected a declaration after decorator"));
                }
                if self.at_identifier("return")
                    || self.at_identifier("break")
                    || self.at_identifier("continue")
                {
                    return Err(Diagnostic::new(
                        "E000121",
                        "`return`, `break`, and `continue` are not valid at module scope",
                        Some(self.peek().span),
                    ));
                }
                match self.statement()? {
                    Statement::Binding(binding) => module.items.push(Item::Binding(binding)),
                    Statement::Expression(expression) => {
                        module.items.push(Item::Expression(expression))
                    }
                    Statement::Return { .. }
                    | Statement::Assert { .. }
                    | Statement::If { .. }
                    | Statement::Match { .. } => {
                        unreachable!("module parsing only requests simple statements")
                    }
                }
            }
            if !self.at(&TokenKind::Newline)
                && !self.at(&TokenKind::Comma)
                && !self.at(&TokenKind::Eof)
            {
                return Err(self.error("expected a newline or comma after declaration"));
            }
            self.separators();
        }
        Ok(module)
    }

    fn decorators(&mut self) -> Result<Vec<Decorator>, Diagnostic> {
        let mut decorators = Vec::new();
        while self.take(&TokenKind::At).is_some() {
            let (name, name_span) = self.identifier("expected an attribute name after `@`")?;
            let mut arguments = Vec::new();
            let mut end = name_span.end;
            if self.take(&TokenKind::LeftParen).is_some() {
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        let start = self.peek().span;
                        let argument_name = if matches!(self.peek().kind, TokenKind::Identifier(_))
                            && self
                                .tokens
                                .get(self.cursor + 1)
                                .is_some_and(|token| token.kind == TokenKind::Equal)
                        {
                            let name = self.identifier("expected an attribute argument name")?.0;
                            self.expect(
                                &TokenKind::Equal,
                                "expected `=` after attribute argument name",
                            )?;
                            Some(name)
                        } else {
                            None
                        };
                        let value_token = self.next();
                        let value = match value_token.kind {
                            TokenKind::String(value) => DecoratorValue::String(value),
                            TokenKind::Integer(value) => DecoratorValue::Integer(value),
                            TokenKind::Identifier(value) if value == "true" => {
                                DecoratorValue::Boolean(true)
                            }
                            TokenKind::Identifier(value) if value == "false" => {
                                DecoratorValue::Boolean(false)
                            }
                            TokenKind::Identifier(value) => DecoratorValue::Name(value),
                            _ => {
                                return Err(Diagnostic::new(
                                    "E000120",
                                    "expected an attribute value",
                                    Some(value_token.span),
                                ))
                            }
                        };
                        arguments.push(DecoratorArgument {
                            name: argument_name,
                            value,
                            span: Span::new(start.source, start.start, value_token.span.end),
                        });
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                end = self
                    .expect(&TokenKind::RightParen, "expected `)` after attribute")?
                    .span
                    .end;
            }
            decorators.push(Decorator {
                name,
                arguments,
                span: Span::new(name_span.source, name_span.start, end),
            });
            if self.take(&TokenKind::Newline).is_none() {
                break;
            }
            while self.take(&TokenKind::Newline).is_some() {}
        }
        Ok(decorators)
    }

    fn function_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<FunctionDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a function name")?;
        let type_parameters = self.type_parameters()?;
        self.expect(&TokenKind::LeftParen, "expected `(` after function name")?;
        let mut parameters = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let (parameter_name, parameter_span) =
                    self.identifier("expected a parameter name")?;
                self.expect(&TokenKind::Colon, "expected `:` after parameter")?;
                let annotation = self.type_annotation()?;
                parameters.push(FunctionParameter {
                    name: parameter_name,
                    span: Span::new(
                        parameter_span.source,
                        parameter_span.start,
                        annotation.span.end,
                    ),
                    annotation,
                });
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        let close = self
            .expect(&TokenKind::RightParen, "expected `)` after parameters")?
            .span;
        let result = if self.take(&TokenKind::Arrow).is_some() {
            self.type_annotation()?
        } else {
            TypeAnnotation::named("unit", vec![], close)
        };
        let mut end = result.span.end;
        let body = if self.take(&TokenKind::Colon).is_some() {
            let (statements, block_end) = self.indented_block("function")?;
            end = block_end;
            Some(statements)
        } else {
            None
        };
        Ok(FunctionDeclaration {
            decorators,
            name,
            type_parameters,
            parameters,
            span: Span::new(start.source, start.start, end),
            result,
            body,
        })
    }

    fn test_declaration(&mut self) -> Result<TestDeclaration, Diagnostic> {
        let start = self.next().span;
        let mut modes = Vec::new();
        if self.at_identifier("with") {
            self.next();
            loop {
                modes.push(self.identifier("expected a test mode after `with`")?.0);
                if !self.at_identifier("and") {
                    break;
                }
                self.next();
            }
        }
        let name = match self.peek().kind.clone() {
            TokenKind::String(name) => {
                self.next();
                Some(name)
            }
            _ => None,
        };
        self.expect(&TokenKind::Colon, "expected `:` after test declaration")?;
        let (body, compiler_cases, end) = if modes.iter().any(|mode| mode == "compiler") {
            let (body, cases, end) = self.compiler_test_block()?;
            (body, cases, end)
        } else {
            let (body, end) = self.indented_block("test")?;
            (body, Vec::new(), end)
        };
        Ok(TestDeclaration {
            name,
            modes,
            body,
            compiler_cases,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn compiler_test_block(
        &mut self,
    ) -> Result<(Vec<Statement>, Vec<CompilerTestCase>, u32), Diagnostic> {
        self.expect(
            &TokenKind::Newline,
            "expected a newline after compiler test header",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(
            &TokenKind::Indent,
            "expected an indented compiler test body",
        )?;
        let mut body = Vec::new();
        let mut cases = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if self.at_identifier("accept") || self.at_identifier("reject") {
                let token = self.next();
                let expectation = match &token.kind {
                    TokenKind::Identifier(value) if value == "accept" => {
                        CompilerExpectation::Accept
                    }
                    _ => CompilerExpectation::Reject,
                };
                let diagnostic_name = if !self.at(&TokenKind::Colon) {
                    Some(self.identifier("expected a diagnostic binding or `:`")?.0)
                } else {
                    None
                };
                self.expect(&TokenKind::Colon, "expected `:` after compiler expectation")?;
                let (case_body, end) = self.indented_block("compiler expectation")?;
                cases.push(CompilerTestCase {
                    expectation,
                    diagnostic_name,
                    body: case_body,
                    span: Span::new(token.span.source, token.span.start, end),
                });
            } else {
                body.push(self.block_statement()?);
                if !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                    return Err(self.error("expected a newline after compiler test assertion"));
                }
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of compiler test body")?
            .span
            .end;
        Ok((body, cases, end))
    }

    fn indented_block(&mut self, owner: &str) -> Result<(Vec<Statement>, u32), Diagnostic> {
        self.expect(
            &TokenKind::Newline,
            &format!("expected a newline after {owner} header"),
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(
            &TokenKind::Indent,
            &format!("expected an indented {owner} body"),
        )?;
        let mut statements = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let compound = self.at_identifier("if") || self.at_identifier("match");
            if self.at_identifier("pass") {
                self.next();
            } else {
                statements.push(self.block_statement()?);
            }
            if !compound && !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error(&format!("expected a newline after {owner} statement")));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, &format!("expected end of {owner} body"))?
            .span
            .end;
        Ok((statements, end))
    }

    fn block_statement(&mut self) -> Result<Statement, Diagnostic> {
        if self.at_identifier("return") {
            let start = self.next().span;
            let value = if self.at(&TokenKind::Newline) || self.at(&TokenKind::Dedent) {
                None
            } else {
                Some(self.expression(0)?)
            };
            let end = value.as_ref().map_or(start.end, |value| value.span.end);
            return Ok(Statement::Return {
                value,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("assert") {
            let start = self.next().span;
            self.expect(&TokenKind::LeftParen, "expected `(` after `assert`")?;
            let condition = self.expression(0)?;
            let message = if self.take(&TokenKind::Comma).is_some() {
                Some(self.expression(0)?)
            } else {
                None
            };
            let end = self
                .expect(&TokenKind::RightParen, "expected `)` after assertion")?
                .span
                .end;
            return Ok(Statement::Assert {
                condition,
                message,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("if") {
            let start = self.next().span;
            let condition = self.expression(0)?;
            self.expect(&TokenKind::Colon, "expected `:` after condition")?;
            let (then_block, mut end) = self.indented_block("if")?;
            let else_block = if self.at_identifier("else") {
                self.next();
                self.expect(&TokenKind::Colon, "expected `:` after `else`")?;
                let (body, block_end) = self.indented_block("else")?;
                end = block_end;
                body
            } else {
                Vec::new()
            };
            return Ok(Statement::If {
                condition,
                then_block,
                else_block,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("match") {
            return self.match_statement();
        }
        if self.at_identifier("break") || self.at_identifier("continue") {
            return Err(self.error("loop control is not implemented yet"));
        }
        self.statement()
    }

    fn match_statement(&mut self) -> Result<Statement, Diagnostic> {
        let start = self.next().span;
        let subject = self.expression(0)?;
        self.expect(&TokenKind::Colon, "expected `:` after match expression")?;
        self.expect(&TokenKind::Newline, "expected a newline after match header")?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected indented match cases")?;
        self.separators();
        let mut cases = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if !self.at_identifier("case") {
                return Err(self.error("match arms must start with `case`"));
            }
            let case_start = self.next().span;
            let pattern_start = self.cursor;
            let (first, _) = self.identifier("expected a case binding, `_`, or type")?;
            let (binding, annotation) = if self.at(&TokenKind::Colon) {
                self.next();
                let binding = (first != "_").then_some(first);
                let annotation = if self.at(&TokenKind::Newline) {
                    None
                } else {
                    let annotation = self.type_annotation()?;
                    self.expect(&TokenKind::Colon, "expected `:` after the case type")?;
                    Some(annotation)
                };
                (binding, annotation)
            } else {
                self.cursor = pattern_start;
                let annotation = self.type_annotation()?;
                let (name, _) = self.identifier("expected a binding after the case type")?;
                self.expect(&TokenKind::Colon, "expected `:` after the case binding")?;
                ((name != "_").then_some(name), Some(annotation))
            };
            let (body, end) = self.indented_block("case")?;
            cases.push(MatchCase {
                binding,
                annotation,
                body,
                span: Span::new(case_start.source, case_start.start, end),
            });
            self.separators();
        }
        if cases.is_empty() {
            return Err(self.error("a match requires at least one case"));
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of match")?
            .span
            .end;
        Ok(Statement::Match {
            subject,
            cases,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn type_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<TypeDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, name_span) = self.identifier("expected a type name")?;
        let type_parameters = self.type_parameters()?;
        let definition = if self.take(&TokenKind::Equal).is_some() {
            Some(self.type_annotation()?)
        } else {
            None
        };
        let end = definition
            .as_ref()
            .map_or(name_span.end, |definition| definition.span.end);
        Ok(TypeDeclaration {
            decorators,
            name,
            type_parameters,
            definition,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn import_declaration(&mut self) -> Result<ImportDeclaration, Diagnostic> {
        let start = self.next().span;
        let subject_token = self.next();
        let subject = match subject_token.kind {
            TokenKind::Identifier(name) => ImportSubject::Name(name),
            TokenKind::String(locator) => ImportSubject::Locator(locator),
            _ => {
                return Err(Diagnostic::new(
                    "E000118",
                    "expected an import name or locator string",
                    Some(subject_token.span),
                ))
            }
        };
        let mut end = subject_token.span.end;
        let source = if self.at_identifier("from") {
            self.next();
            let (source, span) = self.identifier("expected an import source after `from`")?;
            end = span.end;
            Some(source)
        } else {
            None
        };
        let alias = if self.at_identifier("as") {
            self.next();
            let (alias, span) = self.identifier("expected an import alias")?;
            end = span.end;
            Some(alias)
        } else {
            None
        };
        Ok(ImportDeclaration {
            subject,
            source,
            alias,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn trait_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<TraitDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a trait name")?;
        let type_parameters = self.type_parameters()?;
        self.expect(&TokenKind::Colon, "expected `:` after trait name")?;
        let mut bases = Vec::new();
        if !self.at(&TokenKind::Newline) {
            loop {
                bases.push(self.type_annotation()?);
                if self.take(&TokenKind::Plus).is_none() {
                    break;
                }
            }
            self.expect(&TokenKind::Colon, "expected `:` after base traits")?;
        }
        self.expect(&TokenKind::Newline, "expected a newline after trait header")?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented trait body")?;
        let mut properties = Vec::new();
        let mut operators = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if self.at_identifier("property") {
                properties.push(self.property()?);
            } else if self.at_identifier("operator") {
                operators.push(self.operator()?);
            } else if self.at_identifier("pass") {
                self.next();
            } else {
                return Err(self.error("expected `property`, `operator`, or `pass` in trait body"));
            }
            if !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error("expected a newline after trait member"));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of trait body")?
            .span;
        Ok(TraitDeclaration {
            decorators,
            name,
            type_parameters,
            bases,
            properties,
            operators,
            span: Span::new(start.source, start.start, end.end),
        })
    }

    fn type_parameters(&mut self) -> Result<Vec<String>, Diagnostic> {
        let mut type_parameters = Vec::new();
        if self.take(&TokenKind::LeftBracket).is_some() {
            loop {
                type_parameters.push(self.identifier("expected a type parameter")?.0);
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.expect(
                &TokenKind::RightBracket,
                "expected `]` after type parameters",
            )?;
        }
        Ok(type_parameters)
    }

    fn property(&mut self) -> Result<PropertyDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a property name")?;
        self.expect(&TokenKind::Colon, "expected `:` after property name")?;
        let annotation = self.type_annotation()?;
        let default = if self.take(&TokenKind::Equal).is_some() {
            Some(self.expression(0)?)
        } else {
            None
        };
        let end = default
            .as_ref()
            .map_or(annotation.span.end, |expression| expression.span.end);
        Ok(PropertyDeclaration {
            name,
            annotation,
            default,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn operator(&mut self) -> Result<OperatorDeclaration, Diagnostic> {
        let start = self.next().span;
        let operator_token = self.next();
        let operator = operator_syntax(&operator_token.kind).ok_or_else(|| {
            Diagnostic::new(
                "E000117",
                "expected an operator name",
                Some(operator_token.span),
            )
        })?;
        self.expect(&TokenKind::LeftParen, "expected `(` after operator")?;
        let mut parameters = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let (name, span) = self.identifier("expected an operator parameter")?;
                self.expect(&TokenKind::Colon, "expected `:` after parameter")?;
                let annotation = self.type_annotation()?;
                parameters.push(OperatorParameter {
                    name,
                    annotation,
                    span,
                });
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RightParen, "expected `)` after parameters")?;
        self.expect(&TokenKind::Arrow, "expected `->` after operator parameters")?;
        let result = self.type_annotation()?;
        Ok(OperatorDeclaration {
            operator,
            parameters,
            span: Span::new(start.source, start.start, result.span.end),
            result,
        })
    }

    fn binding(&mut self) -> Result<Binding, Diagnostic> {
        if self.looks_like_prefix_typed_binding() {
            let annotation = self.type_annotation()?;
            let start = annotation.span;
            let (name, _) = self.identifier("expected a binding name after its type")?;
            self.expect(&TokenKind::Equal, "expected `=` after binding name")?;
            let value = self.expression(0)?;
            return Ok(Binding {
                name,
                annotation: Some(annotation),
                span: Span::new(start.source, start.start, value.span.end),
                value,
                update: false,
            });
        }
        let (name, name_span) = self.identifier("expected a binding name")?;
        let compound = match self.peek().kind {
            TokenKind::PlusEqual => Some(BinaryOperator::Add),
            TokenKind::MinusEqual => Some(BinaryOperator::Subtract),
            TokenKind::StarEqual => Some(BinaryOperator::Multiply),
            TokenKind::SlashEqual => Some(BinaryOperator::Divide),
            TokenKind::PercentEqual => Some(BinaryOperator::Remainder),
            _ => None,
        };
        if let Some(operator) = compound {
            self.next();
            let right = self.expression(0)?;
            let left = Expression {
                kind: ExpressionKind::Name(name.clone()),
                span: name_span,
            };
            let value = Expression {
                span: Span::new(name_span.source, name_span.start, right.span.end),
                kind: ExpressionKind::Binary {
                    operator,
                    left: Box::new(left),
                    right: Box::new(right),
                },
            };
            return Ok(Binding {
                name,
                annotation: None,
                span: value.span,
                value,
                update: true,
            });
        }
        let inferred = self.take(&TokenKind::ColonEqual).is_some();
        let annotation = if !inferred && self.take(&TokenKind::Colon).is_some() {
            Some(self.type_annotation()?)
        } else {
            None
        };
        if !inferred {
            self.expect(&TokenKind::Equal, "expected `=` or `:=` after binding name")?;
        }
        let value = self.expression(0)?;
        Ok(Binding {
            name,
            annotation,
            span: Span::new(name_span.source, name_span.start, value.span.end),
            value,
            update: false,
        })
    }

    fn statement(&mut self) -> Result<Statement, Diagnostic> {
        if self.looks_like_binding() {
            Ok(Statement::Binding(self.binding()?))
        } else {
            Ok(Statement::Expression(self.expression(0)?))
        }
    }

    fn looks_like_binding(&self) -> bool {
        self.looks_like_prefix_typed_binding()
            || (matches!(self.peek().kind, TokenKind::Identifier(_))
                && self.tokens.get(self.cursor + 1).is_some_and(|token| {
                    matches!(
                        token.kind,
                        TokenKind::Colon
                            | TokenKind::ColonEqual
                            | TokenKind::Equal
                            | TokenKind::PlusEqual
                            | TokenKind::MinusEqual
                            | TokenKind::StarEqual
                            | TokenKind::SlashEqual
                            | TokenKind::PercentEqual
                    )
                }))
    }

    fn looks_like_prefix_typed_binding(&self) -> bool {
        let mut trial = Parser {
            tokens: self.tokens,
            cursor: self.cursor,
        };
        trial.type_annotation().is_ok()
            && matches!(trial.peek().kind, TokenKind::Identifier(_))
            && trial
                .tokens
                .get(trial.cursor + 1)
                .is_some_and(|token| token.kind == TokenKind::Equal)
    }

    fn expression(&mut self, minimum_precedence: u8) -> Result<Expression, Diagnostic> {
        let mut expression = self.unary()?;
        while let Some(operator) = binary_operator(&self.peek().kind) {
            let precedence = precedence(operator);
            if precedence < minimum_precedence {
                break;
            }
            self.next();
            let right_precedence = if operator == BinaryOperator::Power {
                precedence
            } else {
                precedence + 1
            };
            let right = self.expression(right_precedence)?;
            let span = Span::new(
                expression.span.source,
                expression.span.start,
                right.span.end,
            );
            expression = Expression {
                kind: ExpressionKind::Binary {
                    operator,
                    left: Box::new(expression),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(expression)
    }

    fn unary(&mut self) -> Result<Expression, Diagnostic> {
        let operator = match &self.peek().kind {
            TokenKind::Plus => Some(UnaryOperator::Positive),
            TokenKind::Minus => Some(UnaryOperator::Negative),
            TokenKind::Identifier(value) if value == "not" => Some(UnaryOperator::Not),
            TokenKind::Identifier(value) if value == "move" => Some(UnaryOperator::Move),
            _ => None,
        };
        if let Some(operator) = operator {
            let start = self.next().span;
            let operand = self.unary()?;
            return Ok(Expression {
                span: Span::new(start.source, start.start, operand.span.end),
                kind: ExpressionKind::Unary {
                    operator,
                    operand: Box::new(operand),
                },
            });
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expression, Diagnostic> {
        let mut expression = self.primary()?;
        loop {
            if self.take(&TokenKind::Dot).is_some() {
                let (name, member_span) = self.identifier("expected a member name after `.`")?;
                let expression_span = expression.span;
                expression = Expression {
                    kind: ExpressionKind::Member {
                        object: Box::new(expression),
                        name,
                    },
                    span: Span::new(
                        expression_span.source,
                        expression_span.start,
                        member_span.end,
                    ),
                };
            } else if self.take(&TokenKind::LeftParen).is_some() {
                let mut arguments = Vec::new();
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        arguments.push(self.expression(0)?);
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                let end = self
                    .expect(&TokenKind::RightParen, "expected `)` after arguments")?
                    .span
                    .end;
                let span = Span::new(expression.span.source, expression.span.start, end);
                expression = Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(expression),
                        arguments,
                    },
                    span,
                };
            } else {
                break;
            }
        }
        Ok(expression)
    }

    fn primary(&mut self) -> Result<Expression, Diagnostic> {
        let token = self.next();
        let kind = match token.kind {
            TokenKind::Integer(value) => ExpressionKind::Literal(Literal::Integer(value)),
            TokenKind::Float(value) => ExpressionKind::Literal(Literal::Float(value)),
            TokenKind::Character(value) => ExpressionKind::Literal(Literal::Character(value)),
            TokenKind::String(value) => ExpressionKind::Literal(Literal::String(value)),
            TokenKind::FormattedString(value) => {
                return formatted_string_expression(&value, token.span)
            }
            TokenKind::Identifier(value) if value == "true" => {
                ExpressionKind::Literal(Literal::Boolean(true))
            }
            TokenKind::Identifier(value) if value == "false" => {
                ExpressionKind::Literal(Literal::Boolean(false))
            }
            TokenKind::Identifier(value) if value == "None" => {
                ExpressionKind::Literal(Literal::None)
            }
            TokenKind::Identifier(name) => ExpressionKind::Name(name),
            TokenKind::LeftParen => {
                let expression = self.expression(0)?;
                self.expect(&TokenKind::RightParen, "expected `)`")?;
                return Ok(expression);
            }
            _ => {
                return Err(Diagnostic::new(
                    "E000111",
                    "expected a literal or binding name",
                    Some(token.span),
                ))
            }
        };
        Ok(Expression {
            kind,
            span: token.span,
        })
    }

    fn type_annotation(&mut self) -> Result<TypeAnnotation, Diagnostic> {
        let first = self.type_primary()?;
        if self.take(&TokenKind::Pipe).is_none() {
            return Ok(first);
        }
        let start = first.span;
        let mut members = vec![first];
        loop {
            members.push(self.type_primary()?);
            if self.take(&TokenKind::Pipe).is_none() {
                break;
            }
        }
        let end = members.last().expect("union has members").span.end;
        Ok(TypeAnnotation {
            kind: TypeAnnotationKind::Union(members),
            span: Span::new(start.source, start.start, end),
        })
    }

    fn type_primary(&mut self) -> Result<TypeAnnotation, Diagnostic> {
        let (name, start) = self.identifier("expected a type")?;
        let mut arguments = Vec::new();
        let mut end = start.end;
        if self.take(&TokenKind::LeftBracket).is_some() {
            if !self.at(&TokenKind::RightBracket) {
                loop {
                    arguments.push(self.type_annotation()?);
                    if self.take(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            end = self
                .expect(
                    &TokenKind::RightBracket,
                    "expected `]` after type arguments",
                )?
                .span
                .end;
        }
        Ok(TypeAnnotation::named(
            name,
            arguments,
            Span::new(start.source, start.start, end),
        ))
    }

    fn separators(&mut self) {
        while self.at(&TokenKind::Newline) || self.at(&TokenKind::Comma) {
            self.cursor += 1;
        }
    }

    fn identifier(&mut self, message: &str) -> Result<(String, Span), Diagnostic> {
        let token = self.next();
        match token.kind {
            TokenKind::Identifier(name) => Ok((name, token.span)),
            _ => Err(Diagnostic::new("E000110", message, Some(token.span))),
        }
    }

    fn at_identifier(&self, expected: &str) -> bool {
        matches!(&self.peek().kind, TokenKind::Identifier(value) if value == expected)
    }

    fn at(&self, expected: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(expected)
    }

    fn take(&mut self, expected: &TokenKind) -> Option<Token> {
        self.at(expected).then(|| self.next())
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    fn next(&mut self) -> Token {
        let token = self.tokens[self.cursor].clone();
        self.cursor += 1;
        token
    }

    fn expect(&mut self, expected: &TokenKind, message: &str) -> Result<Token, Diagnostic> {
        if self.at(expected) {
            Ok(self.next())
        } else {
            Err(self.error(message))
        }
    }

    fn error(&self, message: &str) -> Diagnostic {
        Diagnostic::new("E000112", message, Some(self.peek().span))
    }
}

fn formatted_string_expression(value: &str, span: Span) -> Result<Expression, Diagnostic> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let bytes = value.as_bytes();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'{' if bytes.get(cursor + 1) == Some(&b'{') => {
                literal.push('{');
                cursor += 2;
            }
            b'}' if bytes.get(cursor + 1) == Some(&b'}') => {
                literal.push('}');
                cursor += 2;
            }
            b'{' => {
                push_string_part(&mut parts, &mut literal, span);
                let end = interpolation_end(value, cursor + 1).ok_or_else(|| {
                    Diagnostic::new(
                        "E000113",
                        "formatted string interpolation is missing `}`",
                        Some(span),
                    )
                })?;
                let source = value[cursor + 1..end].trim();
                if source.is_empty() {
                    return Err(Diagnostic::new(
                        "E000113",
                        "formatted string interpolation may not be empty",
                        Some(span),
                    ));
                }
                parts.push(parse_interpolation(source, span)?);
                cursor = end + 1;
            }
            b'}' => {
                return Err(Diagnostic::new(
                    "E000113",
                    "single `}` in formatted string; write `}}` for a literal brace",
                    Some(span),
                ))
            }
            _ => {
                let character = value[cursor..]
                    .chars()
                    .next()
                    .expect("cursor is inside formatted string");
                literal.push(character);
                cursor += character.len_utf8();
            }
        }
    }
    push_string_part(&mut parts, &mut literal, span);
    let mut parts = parts.into_iter();
    let Some(mut expression) = parts.next() else {
        return Ok(Expression {
            kind: ExpressionKind::Literal(Literal::String(String::new())),
            span,
        });
    };
    for right in parts {
        expression = Expression {
            kind: ExpressionKind::Binary {
                operator: BinaryOperator::Add,
                left: Box::new(expression),
                right: Box::new(right),
            },
            span,
        };
    }
    Ok(expression)
}

fn push_string_part(parts: &mut Vec<Expression>, literal: &mut String, span: Span) {
    if literal.is_empty() {
        return;
    }
    parts.push(Expression {
        kind: ExpressionKind::Literal(Literal::String(std::mem::take(literal))),
        span,
    });
}

fn interpolation_end(value: &str, start: usize) -> Option<usize> {
    let bytes = value.as_bytes();
    let mut cursor = start;
    let mut nesting = 0u32;
    let mut quote = None;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if let Some(expected) = quote {
            if byte == b'\\' {
                cursor = (cursor + 2).min(bytes.len());
                continue;
            }
            if byte == expected {
                quote = None;
            }
        } else {
            match byte {
                b'\'' | b'"' => quote = Some(byte),
                b'(' | b'[' => nesting += 1,
                b')' | b']' => nesting = nesting.saturating_sub(1),
                b'}' if nesting == 0 => return Some(cursor),
                _ => {}
            }
        }
        cursor += 1;
    }
    None
}

fn parse_interpolation(source: &str, outer_span: Span) -> Result<Expression, Diagnostic> {
    let file = SourceFile {
        id: outer_span.source,
        path: "<formatted-string>".into(),
        text: source.to_owned(),
    };
    let tokens = scan(&file)?;
    let mut parser = Parser {
        tokens: &tokens,
        cursor: 0,
    };
    let expression = parser.expression(0)?;
    if !parser.at(&TokenKind::Eof) {
        return Err(Diagnostic::new(
            "E000113",
            "formatted string interpolation must contain one expression",
            Some(outer_span),
        ));
    }
    Ok(expression)
}

fn operator_syntax(kind: &TokenKind) -> Option<OperatorSyntax> {
    Some(match kind {
        TokenKind::Identifier(value) if value == "and" => OperatorSyntax::And,
        TokenKind::Identifier(value) if value == "or" => OperatorSyntax::Or,
        TokenKind::Identifier(value) if value == "not" => OperatorSyntax::Not,
        TokenKind::Plus => OperatorSyntax::Plus,
        TokenKind::Minus => OperatorSyntax::Minus,
        TokenKind::Star => OperatorSyntax::Multiply,
        TokenKind::Slash => OperatorSyntax::Divide,
        TokenKind::Percent => OperatorSyntax::Remainder,
        TokenKind::Power => OperatorSyntax::Power,
        TokenKind::EqualEqual => OperatorSyntax::Equal,
        TokenKind::NotEqual => OperatorSyntax::NotEqual,
        TokenKind::Less => OperatorSyntax::Less,
        TokenKind::LessEqual => OperatorSyntax::LessEqual,
        TokenKind::Greater => OperatorSyntax::Greater,
        TokenKind::GreaterEqual => OperatorSyntax::GreaterEqual,
        TokenKind::Identifier(value) if value == "in" => OperatorSyntax::Contains,
        _ => return None,
    })
}

fn binary_operator(kind: &TokenKind) -> Option<BinaryOperator> {
    Some(match operator_syntax(kind)? {
        OperatorSyntax::Plus => BinaryOperator::Add,
        OperatorSyntax::Minus => BinaryOperator::Subtract,
        OperatorSyntax::Multiply => BinaryOperator::Multiply,
        OperatorSyntax::Divide => BinaryOperator::Divide,
        OperatorSyntax::Remainder => BinaryOperator::Remainder,
        OperatorSyntax::Power => BinaryOperator::Power,
        OperatorSyntax::Equal => BinaryOperator::Equal,
        OperatorSyntax::NotEqual => BinaryOperator::NotEqual,
        OperatorSyntax::Less => BinaryOperator::Less,
        OperatorSyntax::LessEqual => BinaryOperator::LessEqual,
        OperatorSyntax::Greater => BinaryOperator::Greater,
        OperatorSyntax::GreaterEqual => BinaryOperator::GreaterEqual,
        OperatorSyntax::Contains => BinaryOperator::Contains,
        OperatorSyntax::And => BinaryOperator::And,
        OperatorSyntax::Or => BinaryOperator::Or,
        OperatorSyntax::Not => return None,
    })
}

fn precedence(operator: BinaryOperator) -> u8 {
    match operator {
        BinaryOperator::Or => 1,
        BinaryOperator::And => 2,
        BinaryOperator::Equal
        | BinaryOperator::NotEqual
        | BinaryOperator::Less
        | BinaryOperator::LessEqual
        | BinaryOperator::Greater
        | BinaryOperator::GreaterEqual => 3,
        BinaryOperator::Contains => 3,
        BinaryOperator::Add | BinaryOperator::Subtract => 4,
        BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Remainder => 5,
        BinaryOperator::Power => 6,
    }
}
