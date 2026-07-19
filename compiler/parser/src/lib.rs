#![forbid(unsafe_code)]

use severian_ast::{
    AssertStmt, AssignOp, AssignStmt, AsyncExpr, AwaitExpr, BinaryExpr, BinaryOp, Block, CallArg,
    CallExpr, ChaosAction, ChaosRuleExpr, ClassDecl, CollectionExpr, ConstructorDecl, Decorator,
    DecoratorSymbol, ElseBranch, EnumDecl, EnumVariant, Expr, Field, ForStmt, FunctionDecl, Ident,
    IfStmt, ImportDecl, ImportKind, ImportName, IndexExpr, Item, LetKind, LetStmt,
    ListComprehensionExpr, Literal, MapEntry, MapExpr, MemberExpr, Module, OwnershipExpr,
    OwnershipOp, Parameter, Pattern, ReturnStmt, Span, Stmt, SwitchArm, SwitchStmt, TaskOwner,
    TestBlock, TestMode, TraitDecl, TraitMethod, Type, TypeArg, TypePath, UnaryExpr, UnaryOp,
    UnsafeBlock, WhileStmt,
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
    Parser {
        tokens,
        current: 0,
        test_depth: 0,
        unsafe_depth: 0,
    }
    .parse_module()
}

struct Parser<'tokens> {
    tokens: &'tokens [Token],
    current: usize,
    test_depth: usize,
    unsafe_depth: usize,
}

impl Parser<'_> {
    fn parse_module(mut self) -> Result<Module, ParseError> {
        let start = self.peek().span.start;
        let mut items = Vec::new();
        while !self.at(&TokenKind::Eof) {
            let item = if self.at(&TokenKind::At) {
                let decorators = self.parse_decorators()?;
                if self.at(&TokenKind::Def) {
                    Item::Function(self.parse_function_with_decorators(decorators)?)
                } else if self.at(&TokenKind::Class) {
                    Item::Class(self.parse_class_with_decorators(decorators)?)
                } else {
                    return Err(self.error("decorators require a function or class"));
                }
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

    fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
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

    fn parse_trait(&mut self) -> Result<TraitDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Trait, "`trait`")?.span.start;
        let name = self.expect_identifier("trait name")?;
        self.expect_simple(TokenKind::Colon, "`:` after trait name")?;
        self.expect_simple(TokenKind::Newline, "newline after trait header")?;
        self.expect_simple(TokenKind::Indent, "indented trait body")?;
        let mut methods = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
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
            methods,
        })
    }

    fn parse_class(&mut self) -> Result<ClassDecl, ParseError> {
        self.parse_class_with_decorators(Vec::new())
    }

    fn parse_class_with_decorators(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<ClassDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Class, "`class`")?.span.start;
        let name = self.expect_identifier("class name")?;
        let mut traits = Vec::new();
        if self.take_simple(&TokenKind::Colon).is_some() && !self.at(&TokenKind::Newline) {
            loop {
                traits.push(self.parse_type()?);
                if self.take_simple(&TokenKind::Comma).is_none() {
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
                        body: function.body,
                        tests: function.tests,
                    });
                } else {
                    methods.push(function);
                }
            } else {
                let field_start = self.peek().span.start;
                let field_name = self.expect_identifier("field name")?;
                self.expect_simple(TokenKind::Colon, "`:` after field name")?;
                let ty = self.parse_type()?;
                let default = if self.take_simple(&TokenKind::Equal).is_some() {
                    Some(self.parse_expression()?)
                } else {
                    None
                };
                let end = self
                    .expect_simple(TokenKind::Newline, "newline after field")?
                    .span
                    .end;
                fields.push(Field {
                    span: Span::new(field_start, end),
                    name: field_name,
                    ty: Some(ty),
                    default,
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
            traits,
            fields,
            constructors,
            methods,
        })
    }

    fn parse_import(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.peek().span.start;
        let kind = if self.take_simple(&TokenKind::Import).is_some() {
            let path = self.parse_path()?;
            let alias = if self.take_simple(&TokenKind::As).is_some() {
                Some(self.expect_identifier("import alias")?)
            } else {
                None
            };
            ImportKind::Module { path, alias }
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

    fn parse_path(&mut self) -> Result<Vec<Ident>, ParseError> {
        let mut path = vec![self.expect_identifier("module name")?];
        while self.take_simple(&TokenKind::Dot).is_some() {
            path.push(self.expect_identifier("module path segment")?);
        }
        Ok(path)
    }

    fn parse_typed_global(&mut self) -> Result<Stmt, ParseError> {
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

    fn parse_function(&mut self) -> Result<FunctionDecl, ParseError> {
        self.parse_function_with_decorators(Vec::new())
    }

    fn parse_function_with_decorators(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<FunctionDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Def, "`def`")?.span.start;
        let name = self.expect_identifier("function name")?;
        if self.take_simple(&TokenKind::LeftBracket).is_some() {
            while !self.at(&TokenKind::RightBracket) {
                self.expect_identifier("generic parameter")?;
                if self.take_simple(&TokenKind::Colon).is_some() {
                    self.parse_type()?;
                }
                if self.take_simple(&TokenKind::Plus).is_some() {
                    self.parse_type()?;
                }
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.expect_simple(TokenKind::RightBracket, "`]`")?;
        }
        let params = self.parse_parameters()?;
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
            decorators,
            name,
            params,
            return_type,
            body,
            tests,
        })
    }

    fn parse_decorators(&mut self) -> Result<Vec<Decorator>, ParseError> {
        let mut decorators = Vec::new();
        while self.at(&TokenKind::At) {
            let start = self.expect_simple(TokenKind::At, "`@`")?.span.start;
            let segments = self.parse_path()?;
            let name_start = segments.first().unwrap().span.start;
            let name_end = segments.last().unwrap().span.end;
            self.expect_simple(TokenKind::LeftParen, "`(` after decorator name")?;
            let mut symbols = Vec::new();
            if !self.at(&TokenKind::RightParen) {
                loop {
                    let token = self.advance().clone();
                    let spelling = match token.kind {
                        TokenKind::Identifier(value) => value,
                        TokenKind::Star => "*".into(),
                        TokenKind::Caret => "^".into(),
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
            }
            let end = self
                .expect_simple(TokenKind::RightParen, "`)` after decorator symbols")?
                .span
                .end;
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

    fn parse_parameters(&mut self) -> Result<Vec<Parameter>, ParseError> {
        self.expect_simple(TokenKind::LeftParen, "`(`")?;
        let mut params = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let param_start = self.peek().span.start;
                let param_name = self.expect_identifier("parameter name")?;
                self.expect_simple(TokenKind::Colon, "`:` after parameter name")?;
                let ty = self.parse_type()?;
                let default = if self.take_simple(&TokenKind::Equal).is_some() {
                    Some(self.parse_expression()?)
                } else {
                    None
                };
                let end = default
                    .as_ref()
                    .map_or(ty.span().end, |value| value.span().end);
                params.push(Parameter {
                    span: Span::new(param_start, end),
                    name: param_name,
                    ty: Some(ty),
                    default,
                });
                if self.take_simple(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect_simple(TokenKind::RightParen, "`)`")?;
        Ok(params)
    }

    fn parse_test(&mut self) -> Result<TestBlock, ParseError> {
        let start = self.expect_simple(TokenKind::Test, "`test`")?.span.start;
        let mut modes = Vec::new();
        if self.take_simple(&TokenKind::With).is_some() {
            loop {
                let mode = self.expect_identifier("test mode")?;
                modes.push(match mode.name.as_str() {
                    "property" => TestMode::Property,
                    "bench" => TestMode::Bench,
                    "chaos" => TestMode::Chaos,
                    "integration" => TestMode::Integration,
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
            body,
        })
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let first = self.parse_named_type()?;
        if !self.at(&TokenKind::Pipe) {
            return Ok(first);
        }
        let start = first.span().start;
        let mut alternatives = vec![first];
        while self.take_simple(&TokenKind::Pipe).is_some() {
            alternatives.push(self.parse_named_type()?);
        }
        let end = alternatives.last().unwrap().span().end;
        Ok(Type::Union {
            span: Span::new(start, end),
            alternatives,
        })
    }

    fn parse_named_type(&mut self) -> Result<Type, ParseError> {
        let name = self.expect_identifier("type name")?;
        let start = name.span.start;
        let mut args = Vec::new();
        let mut end = name.span.end;
        if self.take_simple(&TokenKind::LeftBracket).is_some() {
            if !self.at(&TokenKind::RightBracket) {
                loop {
                    let ty = self.parse_type()?;
                    args.push(TypeArg {
                        span: ty.span(),
                        ty: Box::new(ty),
                    });
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
        }
        Ok(Type::Named(TypePath {
            span: Span::new(start, end),
            segments: vec![name],
            args,
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
        if self.at(&TokenKind::While) {
            return self.parse_while().map(Stmt::While);
        }
        if self.at(&TokenKind::For) {
            return self.parse_for().map(Stmt::For);
        }
        if self.at(&TokenKind::Switch) {
            return self.parse_switch().map(Stmt::Switch);
        }
        if self.at(&TokenKind::Unsafe) {
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

    fn parse_if(&mut self) -> Result<IfStmt, ParseError> {
        let start = self.expect_simple(TokenKind::If, "`if`")?.span.start;
        let condition = self.parse_expression()?;
        self.expect_simple(TokenKind::Colon, "`:` after if condition")?;
        let then_block = self.parse_suite("if")?;
        let mut end = then_block.span.end;
        let else_branch = if self.take_simple(&TokenKind::Else).is_some() {
            self.expect_simple(TokenKind::Colon, "`:` after else")?;
            let block = self.parse_suite("else")?;
            end = block.span.end;
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

    fn parse_while(&mut self) -> Result<WhileStmt, ParseError> {
        let start = self.expect_simple(TokenKind::While, "`while`")?.span.start;
        let condition = self.parse_expression()?;
        let setup = if self.take_simple(&TokenKind::With).is_some() {
            let name = self.expect_identifier("while setup binding")?;
            let setup_start = name.span.start;
            self.expect_simple(TokenKind::ChangeableEqual, "`:=` in while setup")?;
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
        self.expect_simple(TokenKind::Colon, "`:` after while")?;
        let body = self.parse_suite("while")?;
        Ok(WhileStmt {
            span: Span::new(start, body.span.end),
            setup,
            condition,
            body,
        })
    }

    fn parse_for(&mut self) -> Result<ForStmt, ParseError> {
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
        self.expect_simple(TokenKind::Colon, "`:` after for")?;
        let body = self.parse_suite("for")?;
        Ok(ForStmt {
            span: Span::new(start, body.span.end),
            pattern,
            iterable,
            body,
        })
    }

    fn parse_switch(&mut self) -> Result<SwitchStmt, ParseError> {
        let start = self
            .expect_simple(TokenKind::Switch, "`switch`")?
            .span
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
            let pattern = self.parse_pattern()?;
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
            self.expect_simple(TokenKind::Colon, "`:` after switch pattern")?;
            let body = self.parse_suite("switch arm")?;
            arms.push(SwitchArm {
                span: Span::new(arm_start, body.span.end),
                source,
                pattern,
                guard,
                body,
            });
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

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
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

    fn parse_suite(&mut self, owner: &str) -> Result<Block, ParseError> {
        self.expect_simple(TokenKind::Newline, &format!("newline after {owner} header"))?;
        self.expect_simple(TokenKind::Indent, &format!("indented {owner} body"))?;
        let body = self.parse_block()?;
        self.expect_simple(TokenKind::Dedent, &format!("end of {owner} body"))?;
        Ok(body)
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_and()?;
        while self.take_simple(&TokenKind::Or).is_some() {
            expression = binary(expression, BinaryOp::Or, self.parse_and()?);
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_equality()?;
        while self.take_simple(&TokenKind::And).is_some() {
            expression = binary(expression, BinaryOp::And, self.parse_equality()?);
        }
        Ok(expression)
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
            } else if self.take_simple(&TokenKind::In).is_some() {
                Some(BinaryOp::In)
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
        let mut expression = self.parse_unary()?;
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
            expression = binary(expression, op, self.parse_unary()?);
        }
        Ok(expression)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
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
                    captures: Vec::new(),
                }));
            }
            let value = self.parse_postfix()?;
            self.expect_simple(TokenKind::With, "`with` after async operation")?;
            let owner_name = self.expect_identifier("task owner")?;
            let owner = if owner_name.name == "runtime" {
                TaskOwner::Runtime
            } else {
                TaskOwner::SelfOwned
            };
            let mut captures = Vec::new();
            while self.take_simple(&TokenKind::And).is_some() {
                captures.push(self.expect_identifier("captured capability")?);
            }
            let end = captures
                .last()
                .map_or(owner_name.span.end, |capture| capture.span.end);
            return Ok(Expr::Async(AsyncExpr {
                span: Span::new(start, end),
                value: Box::new(value),
                owner,
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
                self.parse_postfix()
            }
        }
    }

    fn parse_send(&mut self) -> Result<Expr, ParseError> {
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

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
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
                let index = self.parse_expression()?;
                let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
                expression = Expr::Index(IndexExpr {
                    span: Span::new(start, end),
                    object: Box::new(expression),
                    index: Box::new(index),
                });
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

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
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
            TokenKind::LeftParen => self.parse_parenthesized(token.span.start),
            TokenKind::LeftBracket => self.parse_list(token.span.start),
            TokenKind::LeftBrace => self.parse_braces(token.span.start),
            _ => Err(ParseError {
                span: token.span,
                message: "expected an expression".into(),
            }),
        }
    }

    fn parse_parenthesized(&mut self, start: usize) -> Result<Expr, ParseError> {
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

    fn parse_list(&mut self, start: usize) -> Result<Expr, ParseError> {
        if self.at(&TokenKind::RightBracket) {
            let end = self.advance().span.end;
            return Ok(Expr::List(CollectionExpr {
                span: Span::new(start, end),
                elements: Vec::new(),
            }));
        }
        let first = self.parse_expression()?;
        if self.take_simple(&TokenKind::For).is_some() {
            let variable = self.expect_identifier("comprehension variable")?;
            self.expect_simple(TokenKind::In, "`in`")?;
            let iterable = self.parse_expression()?;
            let condition = if self.take_simple(&TokenKind::If).is_some() {
                Some(Box::new(self.parse_expression()?))
            } else {
                None
            };
            let end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
            return Ok(Expr::ListComprehension(ListComprehensionExpr {
                span: Span::new(start, end),
                element: Box::new(first),
                variable,
                iterable: Box::new(iterable),
                condition,
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

    fn parse_braces(&mut self, start: usize) -> Result<Expr, ParseError> {
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

    fn take_assign_op(&mut self) -> Option<AssignOp> {
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
        std::mem::discriminant(&self.peek_token(offset).kind) == std::mem::discriminant(kind)
    }

    fn peek_token(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.current + offset)
            .unwrap_or_else(|| self.tokens.last().unwrap())
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

fn internal_call(name: &str, span: Span, args: Vec<Expr>) -> Expr {
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
