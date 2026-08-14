use super::*;

impl Parser<'_> {
    pub(super) fn parse_module(mut self) -> Result<Module, ParseError> {
        let start = self.peek().span.start;
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            if self.at(&TokenKind::Unsafe) {
                items.extend(self.parse_unsafe_native_block()?);
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
            } else if self.at(&TokenKind::Native) {
                return Err(self.error(
                    "native declarations cross the host ABI and require an `unsafe:` block",
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
        self.expect_simple(TokenKind::Newline, "newline after trait header")?;
        self.expect_simple(TokenKind::Indent, "indented trait body")?;
        let mut methods = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.take_simple(&TokenKind::Def);
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
            methods,
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
