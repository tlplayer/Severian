use super::*;

impl Parser<'_> {
    pub(super) fn parse_module(mut self) -> Result<Module, ParseError> {
        let start = self.peek().span.start;
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            if self.at(&TokenKind::Unsafe) {
                items.extend(self.parse_unsafe_extern_block()?);
                continue;
            }
            let item = if self.at(&TokenKind::At) {
                let decorators = self.parse_decorators()?;
                if self.at(&TokenKind::Def) {
                    Item::Function(self.parse_function_with_decorators(decorators)?)
                } else if self.at(&TokenKind::Class) {
                    Item::Class(self.parse_class_with_decorators(decorators)?)
                } else {
                    return Err(self.error("decorators require a function or class"));
                }
            } else if self.at(&TokenKind::Extern) {
                return Err(self.error(
                    "extern declarations cross the host ABI and require an `unsafe:` block",
                ));
            } else if self.at(&TokenKind::Def) {
                Item::Function(self.parse_function()?)
            } else if self.at(&TokenKind::Class) {
                Item::Class(self.parse_class()?)
            } else if self.at(&TokenKind::Trait) {
                Item::Trait(self.parse_trait()?)
            } else if self.at(&TokenKind::Enum) {
                Item::Enum(self.parse_enum()?)
            } else if self.at(&TokenKind::Import) || self.at(&TokenKind::From) {
                Item::Import(self.parse_import()?)
            } else if matches!(self.peek().kind, TokenKind::Identifier(_))
                && matches!(self.peek_token(1).kind, TokenKind::Identifier(_))
                && self.peek_kind(2, &TokenKind::Equal)
            {
                Item::Statement(self.parse_typed_global()?)
            } else {
                return Err(self.error("expected a declaration or import"));
            };
            items.push(item);
        }
        Ok(Module {
            span: Span::new(start, self.peek().span.end),
            items,
        })
    }

    pub(super) fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Enum, "`enum`")?.span.start;
        let name = self.expect_identifier("enum name")?;
        self.expect_simple(TokenKind::Colon, "`:` after enum name")?;
        self.expect_simple(TokenKind::Newline, "newline after enum header")?;
        self.expect_simple(TokenKind::Indent, "indented enum variants")?;
        let mut variants = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let variant = self.expect_identifier("enum variant")?;
            let variant_start = variant.span.start;
            let mut fields = Vec::new();
            if self.take_simple(&TokenKind::LeftParen).is_some() {
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        let field_start = self.peek().span.start;
                        let field_name = self.expect_identifier("variant field")?;
                        self.expect_simple(TokenKind::Colon, "`:` after variant field")?;
                        let ty = self.parse_type()?;
                        fields.push(Parameter {
                            span: Span::new(field_start, ty.span().end),
                            name: field_name,
                            ty: Some(ty),
                            default: None,
                        });
                        if self.take_simple(&TokenKind::Comma).is_none() {
                            break;
                        }
                        if self.at(&TokenKind::RightParen) {
                            break;
                        }
                    }
                }
                self.expect_simple(TokenKind::RightParen, "`)` after enum variant")?;
            }
            let end = self
                .expect_simple(TokenKind::Newline, "newline after enum variant")?
                .span
                .end;
            variants.push(EnumVariant {
                span: Span::new(variant_start, end),
                name: variant,
                fields,
            });
        }
        let end = self
            .expect_simple(TokenKind::Dedent, "end of enum")?
            .span
            .end;
        Ok(EnumDecl {
            span: Span::new(start, end),
            name,
            variants,
        })
    }

    pub(super) fn parse_trait(&mut self) -> Result<TraitDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Trait, "`trait`")?.span.start;
        let name = self.expect_identifier("trait name")?;
        let generic_params = self.parse_generic_parameters()?;
        self.expect_simple(TokenKind::Colon, "`:` after trait name")?;
        let mut composed_traits = Vec::new();
        if !self.at(&TokenKind::Newline) {
            loop {
                composed_traits.push(self.parse_type()?);
                if self.take_simple(&TokenKind::Plus).is_none() {
                    break;
                }
            }
            // A closing colon is accepted for the Python-shaped spelling
            // `trait Profile: Time + Memory:`. The terser form without it
            // remains valid and is equivalent.
            self.take_simple(&TokenKind::Colon);
        }
        self.expect_simple(TokenKind::Newline, "newline after trait header")?;
        self.expect_simple(TokenKind::Indent, "indented trait body")?;
        let mut decorators = Vec::new();
        let mut methods = Vec::new();
        let mut operators = Vec::new();
        let mut scoped_behaviors = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if self.at(&TokenKind::At) {
                decorators.extend(self.parse_decorators()?);
                continue;
            }
            let behavior_phase = if self.at(&TokenKind::With) {
                Some(TraitScopedBehaviorPhase::With)
            } else if matches!(&self.peek().kind, TokenKind::Identifier(name) if name == "without")
                && self.peek_kind(1, &TokenKind::LeftParen)
            {
                Some(TraitScopedBehaviorPhase::Without)
            } else {
                None
            };
            if let Some(phase) = behavior_phase {
                let behavior_start = self.advance().span.start;
                let params = self.parse_parameters()?;
                if params.len() != 1 || params[0].name.name != "context" {
                    return Err(ParseError {
                        span: Span::new(
                            behavior_start,
                            params
                                .last()
                                .map_or(behavior_start, |parameter| parameter.span.end),
                        ),
                        message: "trait scoped behavior requires exactly one `context` parameter"
                            .into(),
                    });
                }
                self.expect_simple(TokenKind::Colon, "`:` after scoped behavior")?;
                let body = self.parse_suite(match phase {
                    TraitScopedBehaviorPhase::With => "with",
                    TraitScopedBehaviorPhase::Without => "without",
                })?;
                scoped_behaviors.push(TraitScopedBehavior {
                    span: Span::new(behavior_start, body.span.end),
                    phase,
                    params,
                    body,
                });
                continue;
            }
            if self.take_simple(&TokenKind::Operator).is_some() {
                let operator_start = self.peek().span.start;
                let token = self.advance().clone();
                let symbol = match token.kind {
                    TokenKind::Pipe => "|".into(),
                    TokenKind::Ampersand => "&".into(),
                    TokenKind::Caret => "^".into(),
                    TokenKind::Plus => "+".into(),
                    TokenKind::Minus => "-".into(),
                    TokenKind::Star => "*".into(),
                    TokenKind::Power => "**".into(),
                    TokenKind::Slash => "/".into(),
                    TokenKind::Percent => "%".into(),
                    TokenKind::And => "and".into(),
                    TokenKind::Or => "or".into(),
                    TokenKind::Not => "not".into(),
                    TokenKind::At => "@".into(),
                    TokenKind::Identifier(symbol) => symbol,
                    _ => {
                        return Err(ParseError {
                            span: token.span,
                            message: "expected operator symbol in trait contract".into(),
                        })
                    }
                };
                let params = self.parse_parameters()?;
                let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let end = self
                    .expect_simple(TokenKind::Newline, "newline after trait operator")?
                    .span
                    .end;
                operators.push(TraitOperator {
                    span: Span::new(operator_start, end),
                    symbol,
                    params,
                    return_type,
                });
                continue;
            }
            let explicit_method = self.take_simple(&TokenKind::Def).is_some();
            if !explicit_method && !self.peek_kind(1, &TokenKind::LeftParen) {
                composed_traits.push(self.parse_type()?);
                self.expect_simple(
                    TokenKind::Newline,
                    "newline after composed trait requirement",
                )?;
                continue;
            }
            let method_start = self.peek().span.start;
            let method_name = self.expect_identifier("trait method name")?;
            let params = self.parse_parameters()?;
            let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            let end = self
                .expect_simple(TokenKind::Newline, "newline after trait method")?
                .span
                .end;
            methods.push(TraitMethod {
                span: Span::new(method_start, end),
                name: method_name,
                params,
                return_type,
            });
        }
        let end = self
            .expect_simple(TokenKind::Dedent, "end of trait body")?
            .span
            .end;
        Ok(TraitDecl {
            span: Span::new(start, end),
            name,
            generic_params,
            decorators,
            composed_traits,
            methods,
            operators,
            scoped_behaviors,
        })
    }

    pub(super) fn parse_class(&mut self) -> Result<ClassDecl, ParseError> {
        self.parse_class_with_decorators(Vec::new())
    }

    pub(super) fn parse_class_with_decorators(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<ClassDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Class, "`class`")?.span.start;
        let name = self.expect_identifier("class name")?;
        let generic_params = self.parse_generic_parameters()?;
        let mut traits = Vec::new();
        if self.take_simple(&TokenKind::Colon).is_some() && !self.at(&TokenKind::Newline) {
            loop {
                traits.push(self.parse_type()?);
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(&TokenKind::RightParen) {
                    break;
                }
            }
        }
        self.expect_simple(TokenKind::Newline, "newline after class header")?;
        self.expect_simple(TokenKind::Indent, "indented class body")?;
        let mut fields = Vec::new();
        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if self.at(&TokenKind::Def) || self.at(&TokenKind::At) {
                let function = if self.at(&TokenKind::At) {
                    let decorators = self.parse_decorators()?;
                    self.parse_function_with_decorators(decorators)?
                } else {
                    self.parse_function()?
                };
                if function.name.name == name.name {
                    constructors.push(ConstructorDecl {
                        span: function.span,
                        decorators: function.decorators,
                        name: function.name,
                        params: function.params,
                        contract: function.contract,
                        body: function.body,
                        tests: function.tests,
                    });
                } else {
                    methods.push(function);
                }
            } else {
                let field_start = self.peek().span.start;
                let field_name = self.expect_identifier("field name")?;
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
                let constraints = if self.take_simple(&TokenKind::With).is_some() {
                    self.take_simple(&TokenKind::Newline);
                    self.expect_simple(TokenKind::LeftBrace, "`{` after field `with`")?;
                    self.skip_parenthesized_layout();
                    let mut constraints = Vec::new();
                    while !self.at(&TokenKind::RightBrace) {
                        constraints.push(self.parse_expression()?);
                        if self.take_simple(&TokenKind::Comma).is_none() {
                            break;
                        }
                        self.skip_parenthesized_layout();
                    }
                    self.expect_simple(TokenKind::RightBrace, "`}` after field constraints")?;
                    constraints
                } else {
                    Vec::new()
                };
                if ty.is_none() && default.is_none() {
                    return Err(ParseError {
                        span: field_name.span,
                        message: "an untyped field requires a default value".into(),
                    });
                }
                let end = self
                    .expect_simple(TokenKind::Newline, "newline after field")?
                    .span
                    .end;
                fields.push(Field {
                    span: Span::new(field_start, end),
                    name: field_name,
                    ty,
                    default,
                    constraints,
                });
            }
        }
        let mut end = self
            .expect_simple(TokenKind::Dedent, "end of class body")?
            .span
            .end;
        while self.at(&TokenKind::Test) {
            let test = self.parse_test()?;
            end = test.span.end;
            if let Some(method) = methods.last_mut() {
                method.tests.push(test);
            } else if let Some(constructor) = constructors.last_mut() {
                constructor.tests.push(test);
            } else {
                return Err(self.error("class test requires a method or constructor"));
            }
        }
        Ok(ClassDecl {
            span: Span::new(start, end),
            decorators,
            name,
            generic_params,
            traits,
            fields,
            constructors,
            methods,
        })
    }

    pub(super) fn parse_import(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.peek().span.start;
        let kind = if self.take_simple(&TokenKind::Import).is_some() {
            let local_path = match &self.peek().kind {
                TokenKind::String(path) => Some(path.clone()),
                _ => None,
            };
            if let Some(path) = local_path {
                self.advance();
                let alias = if self.take_simple(&TokenKind::As).is_some() {
                    Some(self.expect_identifier("import alias")?)
                } else {
                    None
                };
                ImportKind::Local { path, alias }
            } else {
                let path = self.parse_path()?;
                let alias = if self.take_simple(&TokenKind::As).is_some() {
                    Some(self.expect_identifier("import alias")?)
                } else {
                    None
                };
                ImportKind::Module { path, alias }
            }
        } else {
            self.expect_simple(TokenKind::From, "`from`")?;
            let module = self.parse_path()?;
            self.expect_simple(TokenKind::Import, "`import`")?;
            let mut names = Vec::new();
            loop {
                let name = self.expect_identifier("imported name")?;
                let start = name.span.start;
                let alias = if self.take_simple(&TokenKind::As).is_some() {
                    Some(self.expect_identifier("import alias")?)
                } else {
                    None
                };
                let end = alias.as_ref().map_or(name.span.end, |alias| alias.span.end);
                names.push(ImportName {
                    span: Span::new(start, end),
                    name,
                    alias,
                });
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            ImportKind::From { module, names }
        };
        let end = self
            .expect_simple(TokenKind::Newline, "newline after import")?
            .span
            .end;
        Ok(ImportDecl {
            span: Span::new(start, end),
            kind,
        })
    }

    pub(super) fn parse_path(&mut self) -> Result<Vec<Ident>, ParseError> {
        let mut path = vec![self.expect_identifier("module name")?];
        while self.take_simple(&TokenKind::Dot).is_some() {
            path.push(self.expect_identifier("module path segment")?);
        }
        Ok(path)
    }
}
