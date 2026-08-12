#![forbid(unsafe_code)]

use severian_ast::{
    AssertStmt, AssignOp, AssignStmt, AsyncExpr, AwaitExpr, BinaryExpr, BinaryOp, Block, CallArg,
    CallExpr, ChaosAction, ChaosRuleExpr, ClassDecl, CollectionExpr, ComprehensionClause,
    ConstructorDecl, Decorator, DecoratorSymbol, DestructureLetStmt, ElseBranch, EnumDecl,
    EnumVariant, Expr, Field, ForStmt, FunctionContract, FunctionDecl, Ident, IfExpr, IfStmt,
    ImportDecl, ImportKind, ImportName, IndexExpr, Item, LambdaBody, LetKind, LetStmt,
    ListComprehensionExpr, Literal, MapComprehensionExpr, MapEntry, MapExpr, MemberExpr, Module,
    OwnershipExpr, OwnershipOp, Parameter, Pattern, ReturnStmt, SetComprehensionExpr, SliceExpr,
    Span, Stmt, SwitchArm, SwitchStmt, TaskOwner, TaskPlacement, TestBlock, TestMode, TraitDecl,
    TraitMethod, Type, TypeArg, TypePath, UnaryExpr, UnaryOp, UnsafeBlock, WhileStmt, WithBlock,
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
        loop_depth: 0,
        task_contexts: Vec::new(),
    }
    .parse_module()
}

struct Parser<'tokens> {
    tokens: &'tokens [Token],
    current: usize,
    test_depth: usize,
    unsafe_depth: usize,
    loop_depth: usize,
    task_contexts: Vec<TaskContext>,
}

#[derive(Clone)]
struct TaskContext {
    owner: TaskOwner,
    placement: TaskPlacement,
    captures: Vec<Ident>,
}

impl Parser<'_> {
    fn parse_module(mut self) -> Result<Module, ParseError> {
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

    fn parse_unsafe_native_block(&mut self) -> Result<Vec<Item>, ParseError> {
        self.expect_simple(TokenKind::Unsafe, "`unsafe`")?;
        self.expect_simple(TokenKind::Colon, "`:` after unsafe")?;
        self.expect_simple(TokenKind::Newline, "newline after unsafe header")?;
        self.expect_simple(TokenKind::Indent, "indented unsafe body")?;

        let mut functions = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if !self.at(&TokenKind::Native) {
                return Err(
                    self.error("module-level `unsafe:` blocks may only declare native functions")
                );
            }
            let start = self.peek().span.start;
            functions.push(self.parse_native_function(start)?);
        }
        self.expect_simple(TokenKind::Dedent, "end of unsafe body")?;

        if functions.is_empty() {
            return Err(self.error("module-level `unsafe:` blocks require a native declaration"));
        }
        while self.at(&TokenKind::Test) {
            functions
                .last_mut()
                .expect("an unsafe native block has at least one declaration")
                .tests
                .push(self.parse_test()?);
        }

        Ok(functions.into_iter().map(Item::Function).collect())
    }

    fn parse_native_function(&mut self, start: usize) -> Result<FunctionDecl, ParseError> {
        self.expect_simple(TokenKind::Native, "`native` inside `unsafe:`")?;
        self.expect_simple(TokenKind::LeftParen, "`(` after `native`")?;
        let symbol = match self.advance().clone() {
            Token {
                kind: TokenKind::String(symbol),
                ..
            } => symbol,
            _ => return Err(self.error("native declarations require a linker symbol string")),
        };
        self.expect_simple(TokenKind::RightParen, "`)` after native linker symbol")?;
        self.expect_simple(TokenKind::Def, "`def` after native linker symbol")?;
        let name = self.expect_identifier("native function name")?;
        self.parse_generic_parameters()?;
        let params = self.parse_parameters()?;
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let end = self
            .expect_simple(TokenKind::Newline, "newline after native declaration")?
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

    fn parse_generic_parameters(&mut self) -> Result<(), ParseError> {
        if self.take_simple(&TokenKind::LeftBracket).is_none() {
            return Ok(());
        }
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
        Ok(())
    }

    fn parse_function_with_decorators(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<FunctionDecl, ParseError> {
        let start = self.expect_simple(TokenKind::Def, "`def`")?.span.start;
        let name = self.expect_identifier("function name")?;
        self.parse_generic_parameters()?;
        let params = self.parse_parameters()?;
        let return_type = if self.take_simple(&TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let contract = if self.take_simple(&TokenKind::With).is_some() {
            Some(self.parse_function_contract()?)
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
            native_symbol: None,
            decorators,
            name,
            params,
            return_type,
            contract,
            body,
            tests,
        })
    }

    fn parse_function_contract(&mut self) -> Result<FunctionContract, ParseError> {
        let start = self
            .expect_simple(TokenKind::LeftBrace, "`{` after function `with`")?
            .span
            .start;
        self.skip_parenthesized_layout();
        let mut requirements = Vec::new();
        let mut capabilities = Vec::new();
        while !self.at(&TokenKind::RightBrace) {
            if self.take_simple(&TokenKind::With).is_some() {
                capabilities.push(self.expect_identifier("function capability")?);
            } else {
                requirements.push(self.parse_expression()?);
            }
            self.take_simple(&TokenKind::Comma);
            self.skip_parenthesized_layout();
        }
        let end = self
            .expect_simple(TokenKind::RightBrace, "`}` after function contract")?
            .span
            .end;
        Ok(FunctionContract {
            span: Span::new(start, end),
            requirements,
            capabilities,
        })
    }

    fn parse_decorators(&mut self) -> Result<Vec<Decorator>, ParseError> {
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

    fn parse_parameters(&mut self) -> Result<Vec<Parameter>, ParseError> {
        self.expect_simple(TokenKind::LeftParen, "`(`")?;
        self.skip_parenthesized_layout();
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

    fn skip_parenthesized_layout(&mut self) {
        while matches!(
            self.peek().kind,
            TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent
        ) {
            self.advance();
        }
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
        let mut segments = vec![name];
        while self.take_simple(&TokenKind::Dot).is_some() {
            segments.push(self.expect_identifier("type path segment")?);
        }
        let mut args = Vec::new();
        let mut end = segments.last().unwrap().span.end;
        if self.take_simple(&TokenKind::LeftBracket).is_some() {
            if !self.at(&TokenKind::RightBracket) {
                loop {
                    let tensor_dimension = segments
                        .first()
                        .is_some_and(|segment| segment.name == "Tensor")
                        && !args.is_empty();
                    if tensor_dimension && matches!(self.peek().kind, TokenKind::Integer(_)) {
                        let TokenKind::Integer(value) = self.peek().kind else {
                            unreachable!()
                        };
                        let token = self.advance().clone();
                        let size = u64::try_from(value)
                            .map_err(|_| self.error("tensor dimensions cannot be negative"))?;
                        args.push(TypeArg::Dimension {
                            span: token.span,
                            size,
                        });
                    } else {
                        let ty = self.parse_type()?;
                        args.push(TypeArg::Type {
                            span: ty.span(),
                            ty: Box::new(ty),
                        });
                    }
                    if self.take_simple(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            end = self.expect_simple(TokenKind::RightBracket, "`]`")?.span.end;
        }
        Ok(Type::Named(TypePath {
            span: Span::new(start, end),
            segments,
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
        if self.at(&TokenKind::Switch) {
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

    fn parse_if(&mut self) -> Result<IfStmt, ParseError> {
        let start = self.expect_simple(TokenKind::If, "`if`")?.span.start;
        self.parse_conditional_branch(start, "if")
    }

    fn parse_conditional_branch(
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

    fn parse_while(&mut self) -> Result<WhileStmt, ParseError> {
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

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
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
        self.parse_conditional()
    }

    fn parse_conditional(&mut self) -> Result<Expr, ParseError> {
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

    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_postfix()?;
        if self.take_simple(&TokenKind::Power).is_some() {
            return Ok(binary(left, BinaryOp::Power, self.parse_unary()?));
        }
        Ok(left)
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

    fn parse_comprehension_clauses(&mut self) -> Result<Vec<ComprehensionClause>, ParseError> {
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

fn task_placement(name: &str) -> Option<TaskPlacement> {
    match name {
        "local" => Some(TaskPlacement::Local),
        "gpu" => Some(TaskPlacement::Gpu),
        "simd" => Some(TaskPlacement::Simd),
        "simt" => Some(TaskPlacement::Simt),
        _ => None,
    }
}

fn is_task_context_symbol(name: &str) -> bool {
    matches!(name, "self" | "runtime") || task_placement(name).is_some()
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
