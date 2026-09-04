use severian_ast::{
    BinaryOperator, Binding, CallArgument, ClassDeclaration, CompilerExpectation, CompilerTestCase,
    Decorator, DecoratorArgument, DecoratorValue, EnumDeclaration, EnumVariant, Expression,
    ExpressionKind, ExtensionDeclaration, FunctionContract, FunctionDeclaration, FunctionParameter,
    GenericConstraint, HookSpecification, ImportDeclaration, ImportSubject, Item, Literal,
    LoopGuard, LoopGuardAction, MatchCase, Module, OperatorDeclaration, OperatorImplementation,
    OperatorParameter, OperatorSyntax, PropertyConstraint, PropertyDeclaration, SelectCase,
    Statement, TaskOwner, TestDeclaration, TraitDeclaration, TypeAnnotation, TypeAnnotationKind,
    TypeDeclaration, UnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_lexer::{scan, Token, TokenKind};
use severian_source::{SourceFile, Span};
use std::collections::BTreeMap;

pub fn parse(tokens: &[Token]) -> Result<Module, Diagnostic> {
    parse_with_max_errors(tokens, 5)
}

pub fn parse_with_max_errors(tokens: &[Token], max_errors: usize) -> Result<Module, Diagnostic> {
    let max_errors = max_errors.max(1);
    let first = match Parser::new(tokens).module() {
        Ok(module) => return Ok(module),
        Err(diagnostic) => diagnostic,
    };
    let mut diagnostics = vec![first];
    let mut recovered = tokens.to_vec();
    let mut omitted = false;
    while let Some(span) = diagnostics.last().and_then(|diagnostic| diagnostic.span) {
        if !suppress_diagnostic_line(&mut recovered, span) {
            break;
        }
        match Parser::new(&recovered).module() {
            Ok(_) => break,
            Err(diagnostic) => {
                if diagnostics.iter().any(|known| {
                    known.code == diagnostic.code
                        && known.message == diagnostic.message
                        && known.span == diagnostic.span
                }) {
                    break;
                }
                if diagnostics.len() == max_errors {
                    omitted = true;
                    break;
                }
                diagnostics.push(diagnostic);
            }
        }
    }
    let mut first = diagnostics.remove(0).with_additional(diagnostics);
    if omitted {
        first = first.with_note(format!(
            "additional diagnostics omitted after {max_errors} errors"
        ));
    }
    Err(first)
}

fn suppress_diagnostic_line(tokens: &mut Vec<Token>, span: Span) -> bool {
    let Some(target) = tokens.iter().position(|token| {
        token.span.source == span.source
            && (token.span.start <= span.start && token.span.end >= span.start
                || token.span.start >= span.start)
    }) else {
        return false;
    };
    let start = tokens[..target]
        .iter()
        .rposition(|token| token.kind == TokenKind::Newline)
        .map_or(0, |index| index + 1);
    let end = tokens[target..]
        .iter()
        .position(|token| token.kind == TokenKind::Newline)
        .map_or(tokens.len(), |index| target + index);
    let removable = (start..end)
        .filter(|index| {
            !matches!(
                tokens[*index].kind,
                TokenKind::Indent | TokenKind::Dedent | TokenKind::Eof
            )
        })
        .collect::<Vec<_>>();
    if removable.is_empty() {
        return false;
    }
    for index in removable.into_iter().rev() {
        tokens.remove(index);
    }
    true
}

struct Parser<'a> {
    tokens: &'a [Token],
    cursor: usize,
    continuation_indents: usize,
    operators: BTreeMap<OperatorSyntax, ParserOperator>,
}

#[derive(Debug, Clone, Copy)]
struct ParserOperator {
    precedence: u8,
    right_associative: bool,
}

impl Parser<'_> {
    fn new(tokens: &[Token]) -> Parser<'_> {
        Parser {
            tokens,
            cursor: 0,
            continuation_indents: 0,
            operators: source_operator_table(tokens),
        }
    }

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
            } else if self.at_identifier("class") {
                module
                    .items
                    .push(Item::Class(self.class_declaration(decorators)?));
                self.separators();
                continue;
            } else if self.at_identifier("extend") {
                module
                    .items
                    .push(Item::Extension(self.extension_declaration(decorators)?));
                self.separators();
                continue;
            } else if self.at_identifier("enum") {
                if !decorators.is_empty() {
                    return Err(self.error("decorators may not precede an enum declaration"));
                }
                module.items.push(Item::Enum(self.enum_declaration()?));
                self.separators();
                continue;
            } else if self.at_identifier("union") {
                if !decorators.is_empty() {
                    return Err(self.error("decorators may not precede a union declaration"));
                }
                module.items.push(Item::Type(self.union_declaration()?));
                self.separators();
                continue;
            } else if self.at_identifier("def") || self.at(&TokenKind::Arrow) {
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
            } else if self.at_identifier("import") || self.at_identifier("from") {
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
                    | Statement::Yield { .. }
                    | Statement::Destructure { .. }
                    | Statement::Defer { .. }
                    | Statement::FieldAssignment { .. }
                    | Statement::IndexAssignment { .. }
                    | Statement::Assert { .. }
                    | Statement::Unsafe { .. }
                    | Statement::Placement { .. }
                    | Statement::Try { .. }
                    | Statement::FallibleElse { .. }
                    | Statement::If { .. }
                    | Statement::While { .. }
                    | Statement::For { .. }
                    | Statement::Break { .. }
                    | Statement::Continue { .. }
                    | Statement::Match { .. }
                    | Statement::Select { .. } => {
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
            self.statement_separators();
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
                        let operator_value = operator_syntax(&value_token.kind)
                            .map(|operator| operator_spelling(operator).to_owned());
                        let value = if let Some(operator) = operator_value {
                            DecoratorValue::Name(operator)
                        } else {
                            match value_token.kind {
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
            if self.at(&TokenKind::Newline) {
                break;
            }
        }
        Ok(decorators)
    }

    fn function_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<FunctionDeclaration, Diagnostic> {
        let introducer = self.next();
        let compile_time = introducer.kind == TokenKind::Arrow;
        let start = introducer.span;
        let (name, _) = self.identifier("expected a function name")?;
        let (mut type_parameters, mut constraints, _) = self.type_parameters()?;
        self.expect(&TokenKind::LeftParen, "expected `(` after function name")?;
        self.line_breaks();
        let mut parameters = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let (parameter_name, parameter_span) =
                    self.identifier("expected a parameter name")?;
                if parameter_name == "self"
                    && !self.at(&TokenKind::Colon)
                    && !self.at(&TokenKind::Ellipsis)
                {
                    self.line_breaks();
                    if self.take(&TokenKind::Comma).is_some() {
                        self.line_breaks();
                        continue;
                    }
                    break;
                }
                let (annotation, variadic) = if self.take(&TokenKind::Colon).is_some() {
                    let annotation = self.type_annotation()?;
                    let variadic = self.take(&TokenKind::Ellipsis).is_some();
                    (annotation, variadic)
                } else if let Some(ellipsis) = self.take(&TokenKind::Ellipsis) {
                    (
                        TypeAnnotation::named("Any", Vec::new(), ellipsis.span),
                        true,
                    )
                } else {
                    return Err(Diagnostic::new(
                        "E000112",
                        "expected `:` or `...` after parameter",
                        Some(self.peek().span),
                    ));
                };
                let default = if self.take(&TokenKind::Equal).is_some() {
                    Some(self.expression(0)?)
                } else {
                    None
                };
                if variadic && default.is_some() {
                    return Err(Diagnostic::new(
                        "E000110",
                        "a variadic parameter cannot have a default value",
                        Some(parameter_span),
                    ));
                }
                parameters.push(FunctionParameter {
                    name: parameter_name,
                    variadic,
                    span: Span::new(
                        parameter_span.source,
                        parameter_span.start,
                        default
                            .as_ref()
                            .map_or(annotation.span.end, |value| value.span.end),
                    ),
                    annotation,
                    default,
                });
                self.line_breaks();
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
                self.line_breaks();
                if variadic && !self.at(&TokenKind::RightParen) {
                    return Err(Diagnostic::new(
                        "E000110",
                        "a variadic parameter must be the final parameter",
                        Some(parameter_span),
                    ));
                }
                if self.at(&TokenKind::RightParen) {
                    break;
                }
            }
        }
        for parameter in &parameters {
            let Some(name) = parameter.annotation.simple_name() else {
                continue;
            };
            if parameter.variadic
                && name.len() == 1
                && name.as_bytes()[0].is_ascii_uppercase()
                && !type_parameters.iter().any(|parameter| parameter == name)
            {
                type_parameters.push(name.to_owned());
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
        let hook_context = self.function_hook_context()?;
        let (additional_constraints, contracts) = if hook_context.is_some() {
            (Vec::new(), Vec::new())
        } else {
            self.function_contracts(&type_parameters)?
        };
        constraints.extend(additional_constraints);
        let mut end = hook_context
            .as_ref()
            .map_or(result.span.end, |(_, span)| span.end);
        let mut hook = hook_context.map(|(context, span)| HookSpecification {
            context,
            with_phase: Vec::new(),
            without_phase: Vec::new(),
            span,
        });
        let body = if self.take(&TokenKind::Colon).is_some() {
            if let Some(specification) = &mut hook {
                let block_end = self.hook_body(specification)?;
                end = block_end;
                Some(Vec::new())
            } else {
                let (statements, block_end) = self.indented_block("function")?;
                end = block_end;
                Some(statements)
            }
        } else {
            None
        };
        Ok(FunctionDeclaration {
            decorators,
            compile_time,
            name,
            type_parameters,
            constraints,
            contracts,
            hook,
            parameters,
            span: Span::new(start.source, start.start, end),
            result,
            body,
        })
    }

    fn function_hook_context(&mut self) -> Result<Option<(String, Span)>, Diagnostic> {
        if !self.at_identifier("with") {
            return Ok(None);
        }
        let Some(Token {
            kind: TokenKind::Identifier(_),
            ..
        }) = self.tokens.get(self.cursor + 1)
        else {
            return Ok(None);
        };
        if !self
            .tokens
            .get(self.cursor + 2)
            .is_some_and(|token| token.kind == TokenKind::Colon)
        {
            return Ok(None);
        }
        let start = self.next().span;
        let (context, context_span) = self.identifier("expected a hook context after `with`")?;
        Ok(Some((
            context,
            Span::new(start.source, start.start, context_span.end),
        )))
    }

    fn hook_body(&mut self, hook: &mut HookSpecification) -> Result<u32, Diagnostic> {
        self.expect(
            &TokenKind::Newline,
            "expected a newline after hook signature",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented hook body")?;
        self.separators();
        let mut saw_with = false;
        let mut saw_without = false;
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let is_with = self.at_identifier("with");
            let is_without = self.at_identifier("without");
            if !is_with && !is_without {
                return Err(self.error("expected `with` or `without` hook phase"));
            }
            let phase = self.next().span;
            let (context, context_span) =
                self.identifier("expected the hook context after phase name")?;
            if context != hook.context {
                return Err(Diagnostic::new(
                    "E000112",
                    format!(
                        "hook phase uses context `{context}`, expected `{}`",
                        hook.context
                    ),
                    Some(context_span),
                ));
            }
            self.expect(&TokenKind::Colon, "expected `:` after hook phase")?;
            let (body, block_end) = self.indented_block(if is_with {
                "with hook phase"
            } else {
                "without hook phase"
            })?;
            if is_with {
                if saw_with {
                    return Err(Diagnostic::new(
                        "E000112",
                        "a hook may only declare one `with` phase",
                        Some(phase),
                    ));
                }
                saw_with = true;
                hook.with_phase = body;
            } else {
                if saw_without {
                    return Err(Diagnostic::new(
                        "E000112",
                        "a hook may only declare one `without` phase",
                        Some(phase),
                    ));
                }
                saw_without = true;
                hook.without_phase = body;
            }
            hook.span = Span::new(hook.span.source, hook.span.start, block_end);
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of hook body")?
            .span
            .end;
        if !saw_with && !saw_without {
            return Err(Diagnostic::new(
                "E000112",
                "a hook body requires a `with` or `without` phase",
                Some(hook.span),
            ));
        }
        hook.span = Span::new(hook.span.source, hook.span.start, end);
        Ok(end)
    }

    fn test_declaration(&mut self) -> Result<TestDeclaration, Diagnostic> {
        let start = self.next().span;
        let mut name = None;
        let mut parameters = Vec::new();
        if matches!(self.peek().kind, TokenKind::Identifier(ref value) if value != "with") {
            name = Some(self.identifier("expected a test name")?.0);
            if self.take(&TokenKind::LeftBracket).is_some() {
                if !self.at(&TokenKind::RightBracket) {
                    loop {
                        self.type_annotation()?;
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RightBracket, "expected `]` after test subject")?;
            }
            if self.take(&TokenKind::LeftParen).is_some() {
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        parameters.push(self.identifier("expected a test parameter")?.0);
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RightParen, "expected `)` after test parameters")?;
            }
        }
        let mut modes = Vec::new();
        if self.at_identifier("with") {
            self.next();
            loop {
                let (mut mode, mode_span) = self.identifier("expected a test mode after `with`")?;
                match mode.as_str() {
                    "timeout" => {
                        self.expect(&TokenKind::LeftParen, "expected `(` after `timeout`")?;
                        let deadline = self.expression(0)?;
                        self.expect(
                            &TokenKind::RightParen,
                            "expected `)` after timeout deadline",
                        )?;
                        let ExpressionKind::Literal(severian_ast::Literal::Measured {
                            magnitude,
                            suffix,
                        }) = deadline.kind
                        else {
                            return Err(Diagnostic::new(
                                "E000112",
                                "a test timeout requires a duration literal",
                                Some(mode_span),
                            ));
                        };
                        mode = format!("timeout:{magnitude}{suffix}");
                    }
                    "repeat" => {
                        self.expect(&TokenKind::LeftParen, "expected `(` after `repeat`")?;
                        let count = self.expression(0)?;
                        self.expect(&TokenKind::RightParen, "expected `)` after repeat count")?;
                        let ExpressionKind::Literal(severian_ast::Literal::Integer(count)) =
                            count.kind
                        else {
                            return Err(Diagnostic::new(
                                "E000112",
                                "a test repeat count requires an integer literal",
                                Some(mode_span),
                            ));
                        };
                        mode = format!("repeat:{count}");
                    }
                    "skip" => {
                        self.expect(&TokenKind::LeftParen, "expected `(` after `skip`")?;
                        let reason = self.expression(0)?;
                        self.expect(&TokenKind::RightParen, "expected `)` after skip reason")?;
                        let ExpressionKind::Literal(severian_ast::Literal::String(reason)) =
                            reason.kind
                        else {
                            return Err(Diagnostic::new(
                                "E000112",
                                "a skipped test requires a string reason",
                                Some(mode_span),
                            ));
                        };
                        mode = format!("skip:{reason}");
                    }
                    _ => {}
                }
                modes.push(mode);
                if self.at_identifier("and") || self.at(&TokenKind::Comma) {
                    self.next();
                } else {
                    break;
                }
            }
        }
        if let TokenKind::String(quoted) = self.peek().kind.clone() {
            self.next();
            name = Some(quoted);
        }
        let mut cases = Vec::new();
        let mut matrix = false;
        if modes.iter().any(|mode| mode == "cases") {
            matrix = if self.at_identifier("with") {
                self.next();
                true
            } else {
                false
            };
            self.line_breaks();
            self.expect(&TokenKind::LeftBrace, "expected `{` after `with cases`")?;
            let multiline = self.take(&TokenKind::Newline).is_some();
            while self.take(&TokenKind::Newline).is_some() {}
            if multiline {
                self.expect(&TokenKind::Indent, "expected indented test cases")?;
            }
            self.separators();
            let mut axes = Vec::new();
            while !self.at(&TokenKind::RightBrace)
                && !self.at(&TokenKind::Dedent)
                && !self.at(&TokenKind::Eof)
            {
                if matrix {
                    let (parameter, parameter_span) =
                        self.identifier("expected a matrix parameter")?;
                    if !self.at_identifier("in") {
                        return Err(Diagnostic::new(
                            "E000112",
                            "expected `in` after matrix parameter",
                            Some(parameter_span),
                        ));
                    }
                    self.next();
                    let values = self.expression(0)?;
                    let ExpressionKind::Set(values) = values.kind else {
                        return Err(Diagnostic::new(
                            "E000112",
                            "a test matrix axis requires a set of values",
                            Some(values.span),
                        ));
                    };
                    if values.is_empty() {
                        return Err(Diagnostic::new(
                            "E000112",
                            "a test matrix axis cannot be empty",
                            Some(parameter_span),
                        ));
                    }
                    axes.push((parameter, values));
                } else {
                    let value = self.expression(0)?;
                    let ExpressionKind::Tuple(values) = value.kind else {
                        return Err(Diagnostic::new(
                            "E000112",
                            "each parameterized test case must be a tuple",
                            Some(value.span),
                        ));
                    };
                    cases.push(values);
                }
                self.take(&TokenKind::Comma);
                self.separators();
            }
            if multiline {
                self.expect(&TokenKind::Dedent, "expected end of test cases")?;
            }
            self.expect(&TokenKind::RightBrace, "expected `}` after test cases")?;
            if matrix {
                if !parameters.is_empty() {
                    return Err(Diagnostic::new(
                        "E000112",
                        "a test matrix declares its parameters inside the case block",
                        Some(self.peek().span),
                    ));
                }
                cases.push(Vec::new());
                for (parameter, values) in axes {
                    parameters.push(parameter);
                    cases = cases
                        .iter()
                        .flat_map(|case| {
                            values.iter().cloned().map(|value| {
                                let mut expanded = case.clone();
                                expanded.push(value);
                                expanded
                            })
                        })
                        .collect();
                }
            }
        }
        if modes.iter().any(|mode| mode == "differential") {
            self.line_breaks();
            self.expect(
                &TokenKind::LeftBrace,
                "expected `{` after `with differential`",
            )?;
            self.line_breaks();
            if self.at(&TokenKind::Indent) {
                self.next();
            }
            self.separators();
            while !self.at(&TokenKind::RightBrace)
                && !self.at(&TokenKind::Dedent)
                && !self.at(&TokenKind::Eof)
            {
                self.identifier("expected a differential backend")?;
                self.take(&TokenKind::Comma);
                self.separators();
            }
            if self.at(&TokenKind::Dedent) {
                self.next();
            }
            self.expect(
                &TokenKind::RightBrace,
                "expected `}` after differential backends",
            )?;
        }
        let (_, contracts) = self.function_contracts(&[])?;
        self.expect(&TokenKind::Colon, "expected `:` after test declaration")?;
        let has_compiler_cases = self.block_has_compiler_expectation();
        if has_compiler_cases && !modes.iter().any(|mode| mode == "compiler") {
            modes.push("compiler".into());
        }
        let (body, compiler_cases, end) = if modes.iter().any(|mode| mode == "compiler") {
            let (body, cases, end) = self.compiler_test_block()?;
            (body, cases, end)
        } else {
            let (body, end) = self.indented_block("test")?;
            (body, Vec::new(), end)
        };
        Ok(TestDeclaration {
            name,
            parameters,
            cases,
            matrix,
            modes,
            contracts,
            body,
            compiler_cases,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn block_has_compiler_expectation(&self) -> bool {
        let mut cursor = self.cursor;
        let mut depth = 0usize;
        while let Some(token) = self.tokens.get(cursor) {
            match &token.kind {
                TokenKind::Indent => depth += 1,
                TokenKind::Dedent => {
                    if depth == 1 {
                        return false;
                    }
                    depth = depth.saturating_sub(1);
                }
                TokenKind::Identifier(name)
                    if depth == 1 && matches!(name.as_str(), "accept" | "reject") =>
                {
                    let mut lookahead = cursor + 1;
                    if self
                        .tokens
                        .get(lookahead)
                        .is_some_and(|token| matches!(token.kind, TokenKind::Identifier(_)))
                    {
                        lookahead += 1;
                    }
                    if self
                        .tokens
                        .get(lookahead)
                        .is_some_and(|token| token.kind == TokenKind::Colon)
                    {
                        return true;
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            cursor += 1;
        }
        false
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
                let (case_items, mut case_body, end) = self.compiler_case_block()?;
                let mut scoped_body = body.clone();
                scoped_body.append(&mut case_body);
                cases.push(CompilerTestCase {
                    expectation,
                    diagnostic_name,
                    items: case_items,
                    body: scoped_body,
                    span: Span::new(token.span.source, token.span.start, end),
                });
            } else if self.at_identifier("import") || self.at_identifier("from") {
                self.import_declaration()?;
            } else {
                body.push(self.block_statement()?);
                if !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                    return Err(self.error("expected a newline after compiler test assertion"));
                }
            }
            self.statement_separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of compiler test body")?
            .span
            .end;
        Ok((Vec::new(), cases, end))
    }

    fn compiler_case_block(&mut self) -> Result<(Vec<Item>, Vec<Statement>, u32), Diagnostic> {
        self.expect(
            &TokenKind::Newline,
            "expected a newline after compiler expectation",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(
            &TokenKind::Indent,
            "expected an indented compiler expectation body",
        )?;
        let mut items = Vec::new();
        let mut statements = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let decorators = if self.at(&TokenKind::At) {
                self.decorators()?
            } else {
                Vec::new()
            };
            if self.at_identifier("def") {
                items.push(Item::Function(self.function_declaration(decorators)?));
            } else if self.at_identifier("class") {
                items.push(Item::Class(self.class_declaration(decorators)?));
            } else if self.at_identifier("extend") {
                items.push(Item::Extension(self.extension_declaration(decorators)?));
            } else if self.at_identifier("enum") && decorators.is_empty() {
                items.push(Item::Enum(self.enum_declaration()?));
            } else if !decorators.is_empty() {
                return Err(self.error("expected `def` after compiler-case decorator"));
            } else {
                statements.push(self.block_statement()?);
            }
            self.statement_separators();
        }
        let end = self
            .expect(
                &TokenKind::Dedent,
                "expected end of compiler expectation body",
            )?
            .span
            .end;
        Ok((items, statements, end))
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
            let compound = self.at_identifier("if")
                || self.at_identifier("when")
                || self.at_identifier("always")
                || self.at_identifier("match")
                || self.at_identifier("select")
                || self.at_identifier("while")
                || self.at_identifier("for")
                || self.at_identifier("with")
                || self.at_identifier("unsafe")
                || self.at_identifier("try");
            if self.at_identifier("pass") {
                self.next();
            } else {
                let statement = self.block_statement()?;
                if owner == "test" && self.take(&TokenKind::Colon).is_some() {
                    let mut lookahead = self.cursor;
                    while self
                        .tokens
                        .get(lookahead)
                        .is_some_and(|token| token.kind == TokenKind::Newline)
                    {
                        lookahead += 1;
                    }
                    if self
                        .tokens
                        .get(lookahead)
                        .is_some_and(|token| token.kind == TokenKind::Indent)
                    {
                        if is_throws_call_statement(&statement) {
                            statements.push(self.structured_throws_statement(statement)?);
                            self.statement_separators();
                            continue;
                        }
                        let (checks, end) = self.indented_block("test step")?;
                        if let Statement::Expression(Expression {
                            kind: ExpressionKind::Fallback { value, fallback },
                            span,
                        }) = statement
                        {
                            if let ExpressionKind::Name(error_binding) = fallback.kind {
                                statements.push(Statement::FallibleElse {
                                    value: *value,
                                    error_binding,
                                    body: checks,
                                    span: Span::new(span.source, span.start, end),
                                });
                            } else {
                                statements.push(Statement::Expression(Expression {
                                    kind: ExpressionKind::Fallback { value, fallback },
                                    span,
                                }));
                                statements.extend(checks);
                            }
                        } else {
                            statements.push(statement);
                            statements.extend(checks);
                        }
                    } else {
                        statements.push(statement);
                    }
                    self.statement_separators();
                    continue;
                }
                statements.push(statement);
            }
            if !compound && !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error(&format!("expected a newline after {owner} statement")));
            }
            self.statement_separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, &format!("expected end of {owner} body"))?
            .span
            .end;
        Ok((statements, end))
    }

    fn structured_throws_statement(
        &mut self,
        statement: Statement,
    ) -> Result<Statement, Diagnostic> {
        let Statement::Expression(Expression {
            kind:
                ExpressionKind::Call {
                    callee: _,
                    arguments,
                },
            span,
        }) = statement
        else {
            unreachable!("structured throws is selected from a throws call")
        };
        let [argument] = arguments.as_slice() else {
            return Err(Diagnostic::new(
                "E000217",
                "`throws` requires exactly one expression",
                Some(span),
            ));
        };

        self.expect(
            &TokenKind::Newline,
            "expected a newline after structured `throws`",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(
            &TokenKind::Indent,
            "expected an indented error pattern after `throws`",
        )?;
        self.separators();

        let pattern_start = self.peek().span;
        let annotation = self.type_annotation()?;
        self.expect(
            &TokenKind::LeftParen,
            "expected `(` after the expected error type",
        )?;
        let (field, field_span) = self.identifier("expected an error field binding")?;
        self.expect(
            &TokenKind::RightParen,
            "expected `)` after the error field binding",
        )?;
        self.expect(
            &TokenKind::Colon,
            "expected `:` after the expected error pattern",
        )?;
        let (mut catch_body, catch_end) = self.indented_block("structured throws error")?;
        self.separators();
        self.expect(&TokenKind::Dedent, "expected end of structured `throws`")?;

        let hidden = format!("__throws_error_{}", span.start);
        catch_body.insert(
            0,
            Statement::Binding(severian_ast::Binding {
                name: field.clone(),
                annotation: None,
                value: Expression {
                    kind: ExpressionKind::Member {
                        object: Box::new(Expression {
                            kind: ExpressionKind::Name(hidden.clone()),
                            span: field_span,
                        }),
                        name: field,
                    },
                    span: field_span,
                },
                mutable: false,
                update: false,
                preserve_error: false,
                span: field_span,
            }),
        );

        let failed_span = Span::new(span.source, span.start, argument.value.span.end);
        let body = vec![
            Statement::Expression(argument.value.clone()),
            Statement::Assert {
                condition: Expression {
                    kind: ExpressionKind::Literal(Literal::Boolean(false)),
                    span: failed_span,
                },
                message: Some(Expression {
                    kind: ExpressionKind::Literal(Literal::String(
                        "expected expression to throw".into(),
                    )),
                    span: failed_span,
                }),
                span: failed_span,
            },
        ];
        Ok(Statement::Try {
            body,
            catch_binding: hidden,
            catch_annotation: Some(annotation),
            catch_body,
            span: Span::new(pattern_start.source, span.start, catch_end),
        })
    }

    fn block_statement(&mut self) -> Result<Statement, Diagnostic> {
        if self.at_identifier("yield") {
            let start = self.next().span;
            let value = self.expression(0)?;
            return Ok(Statement::Yield {
                span: Span::new(start.source, start.start, value.span.end),
                value,
            });
        }
        if self.at_identifier("throws")
            && self
                .tokens
                .get(self.cursor + 1)
                .is_some_and(|token| token.kind == TokenKind::Colon)
        {
            let start = self.next().span;
            self.next();
            let (mut body, end) = self.indented_block("throws")?;
            let span = Span::new(start.source, start.start, end);
            body.push(Statement::Assert {
                condition: Expression {
                    kind: ExpressionKind::Literal(Literal::Boolean(false)),
                    span,
                },
                message: Some(Expression {
                    kind: ExpressionKind::Literal(Literal::String(
                        "expected statement block to throw".into(),
                    )),
                    span,
                }),
                span,
            });
            return Ok(Statement::Try {
                body,
                catch_binding: format!("__throws_error_{}", start.start),
                catch_annotation: None,
                catch_body: Vec::new(),
                span,
            });
        }
        if self.at_identifier("try") {
            let start = self.next().span;
            self.expect(&TokenKind::Colon, "expected `:` after `try`")?;
            let (body, _) = self.indented_block("try")?;
            if !self.at_identifier("catch") {
                return Err(self.error("expected `catch` after `try` body"));
            }
            self.next();
            let pattern_start = self.cursor;
            let (first, _) = self.identifier("expected a catch binding or error type")?;
            let (catch_binding, catch_annotation) = if self.take(&TokenKind::Colon).is_some() {
                if self.at(&TokenKind::Newline) {
                    (first, None)
                } else {
                    let annotation = self.type_annotation()?;
                    self.expect(&TokenKind::Colon, "expected `:` after catch error type")?;
                    (first, Some(annotation))
                }
            } else {
                self.cursor = pattern_start;
                let annotation = self.type_annotation()?;
                let (binding, _) = self.identifier("expected a binding after catch error type")?;
                self.expect(&TokenKind::Colon, "expected `:` after catch binding")?;
                (binding, Some(annotation))
            };
            let (catch_body, end) = self.indented_block("catch")?;
            return Ok(Statement::Try {
                body,
                catch_binding,
                catch_annotation,
                catch_body,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("unsafe") {
            let start = self.next().span;
            self.expect(&TokenKind::Colon, "expected `:` after `unsafe`")?;
            let (body, end) = self.indented_block("unsafe")?;
            return Ok(Statement::Unsafe {
                body,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("drop") {
            let start = self.next().span;
            let object = self.expression(0)?;
            let end = object.span.end;
            let callee = Expression {
                span: Span::new(start.source, object.span.start, end),
                kind: ExpressionKind::Member {
                    object: Box::new(object),
                    name: "drop".into(),
                },
            };
            return Ok(Statement::Expression(Expression {
                kind: ExpressionKind::Call {
                    callee: Box::new(callee),
                    arguments: Vec::new(),
                },
                span: Span::new(start.source, start.start, end),
            }));
        }
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
        if self.at_identifier("defer") {
            let start = self.next().span;
            let expression = self.expression(0)?;
            return Ok(Statement::Defer {
                span: Span::new(start.source, start.start, expression.span.end),
                expression,
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
        if self.at_identifier("when") {
            let start = self.next().span;
            let parenthesized = self.take(&TokenKind::LeftParen).is_some();
            let condition = self.expression(0)?;
            if parenthesized {
                self.expect(
                    &TokenKind::RightParen,
                    "expected `)` after `when` condition",
                )?;
            }
            self.expect(&TokenKind::Colon, "expected `:` after `when` condition")?;
            let (then_block, end) = self.indented_block("when")?;
            return Ok(Statement::If {
                condition,
                then_block,
                else_block: Vec::new(),
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("always") {
            let start = self.next().span;
            self.expect(&TokenKind::Colon, "expected `:` after `always`")?;
            let (then_block, end) = self.indented_block("always")?;
            return Ok(Statement::If {
                condition: Expression {
                    kind: ExpressionKind::Literal(severian_ast::Literal::Boolean(true)),
                    span: start,
                },
                then_block,
                else_block: Vec::new(),
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("if") {
            let start = self.next().span;
            let condition = self.expression(0)?;
            self.expect(&TokenKind::Colon, "expected `:` after condition")?;
            let (then_block, mut end) = self.indented_block("if")?;
            let else_block = if self.at_identifier("else") || self.at_identifier("elif") {
                let (body, block_end) = self.else_clause()?;
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
        if self.at_identifier("while") {
            let start = self.next().span;
            let condition = self.expression(0)?;
            let mut guards = Vec::new();
            let initializer = if self.at_identifier("with") {
                self.next();
                while self.take(&TokenKind::Newline).is_some() {}
                if self.at(&TokenKind::LeftBrace) {
                    guards = self.loop_guards()?;
                    None
                } else {
                    Some(self.binding()?)
                }
            } else {
                None
            };
            self.expect(&TokenKind::Colon, "expected `:` after while condition")?;
            let (body, end) = self.indented_block("while")?;
            return Ok(Statement::While {
                condition,
                initializer,
                guards,
                body,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("for") {
            let start = self.next().span;
            let (binding, _) = self.identifier("expected a loop binding")?;
            let second_binding = if self.take(&TokenKind::Comma).is_some() {
                Some(self.identifier("expected a loop binding after `,`")?.0)
            } else {
                None
            };
            if !self.at_identifier("in") {
                return Err(self.error("expected `in` after loop binding(s)"));
            }
            self.next();
            let iterable = self.expression(0)?;
            let mut placement = None;
            let initializer = if self.at_identifier("with") {
                self.next();
                if matches!(
                    &self.peek().kind,
                    TokenKind::Identifier(policy)
                        if matches!(
                            policy.as_str(),
                            "gpu" | "simd" | "simt" | "parallel" | "tasks" | "distributed"
                        )
                ) {
                    // Execution placement is retained separately from the
                    // optional loop initializer and survives into semantic IR.
                    placement = Some(self.identifier("expected an execution placement")?.0);
                    None
                } else {
                    Some(self.binding()?)
                }
            } else {
                None
            };
            self.expect(&TokenKind::Colon, "expected `:` after for iterable")?;
            let (body, end) = self.indented_block("for")?;
            return Ok(Statement::For {
                binding,
                second_binding,
                iterable,
                initializer,
                placement,
                body,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("with") {
            let start = self.next().span;
            if self.at_identifier("self")
                && self.tokens.get(self.cursor + 1).is_some_and(
                    |token| matches!(&token.kind, TokenKind::Identifier(value) if value == "and"),
                )
                && self.tokens.get(self.cursor + 2).is_some_and(|token| {
                    matches!(
                        &token.kind,
                        TokenKind::Identifier(value)
                            if matches!(value.as_str(), "gpu" | "simd" | "simt")
                    )
                })
            {
                self.next();
                self.next();
                let policy = self
                    .identifier("expected an execution placement after `and`")?
                    .0;
                self.expect(&TokenKind::Colon, "expected `:` after execution placement")?;
                let (body, end) = self.indented_block("execution placement")?;
                return Ok(Statement::Placement {
                    policy,
                    body,
                    span: Span::new(start.source, start.start, end),
                });
            }
            let resource = self.expression(0)?;
            if !self.at_identifier("as") {
                return Err(self.error("expected `as` after context expression"));
            }
            self.next();
            let (binding, _) = self.identifier("expected a context binding after `as`")?;
            self.expect(&TokenKind::Colon, "expected `:` after context binding")?;
            let (body, end) = self.indented_block("with")?;
            let iterable = Expression {
                span: resource.span,
                kind: ExpressionKind::List(vec![resource]),
            };
            return Ok(Statement::For {
                binding,
                second_binding: None,
                iterable,
                initializer: None,
                placement: None,
                body,
                span: Span::new(start.source, start.start, end),
            });
        }
        if self.at_identifier("match") {
            return self.match_statement();
        }
        if self.at_identifier("select") {
            return self.select_statement();
        }
        if self.at_identifier("break") {
            let span = self.next().span;
            return Ok(Statement::Break { span });
        }
        if self.at_identifier("continue") {
            let span = self.next().span;
            return Ok(Statement::Continue { span });
        }
        let statement = self.statement()?;
        if !self.at(&TokenKind::Colon) {
            return Ok(statement);
        }
        match statement {
            Statement::Expression(Expression {
                kind: ExpressionKind::Fallback { value, fallback },
                span,
            }) => {
                if let ExpressionKind::Name(error_binding) = &fallback.kind {
                    self.next();
                    let (body, end) = self.indented_block("fallible else")?;
                    return Ok(Statement::FallibleElse {
                        value: *value,
                        error_binding: error_binding.clone(),
                        body,
                        span: Span::new(span.source, span.start, end),
                    });
                }
                Ok(Statement::Expression(Expression {
                    kind: ExpressionKind::Fallback { value, fallback },
                    span,
                }))
            }
            statement => Ok(statement),
        }
    }

    fn else_clause(&mut self) -> Result<(Vec<Statement>, u32), Diagnostic> {
        let start = self.next().span;
        let conditional = matches!(
            self.tokens.get(self.cursor.saturating_sub(1)).map(|token| &token.kind),
            Some(TokenKind::Identifier(name)) if name == "elif"
        );
        if !conditional && self.take(&TokenKind::Colon).is_some() {
            return self.indented_block("else");
        }
        let condition = self.expression(0)?;
        self.expect(&TokenKind::Colon, "expected `:` after else condition")?;
        let (then_block, mut end) = self.indented_block("else condition")?;
        let else_block = if self.at_identifier("else") || self.at_identifier("elif") {
            let (body, block_end) = self.else_clause()?;
            end = block_end;
            body
        } else {
            Vec::new()
        };
        Ok((
            vec![Statement::If {
                condition,
                then_block,
                else_block,
                span: Span::new(start.source, start.start, end),
            }],
            end,
        ))
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
            let case_start = if self.at_identifier("case") {
                self.next().span
            } else {
                self.peek().span
            };
            let pattern_start = self.cursor;
            let (first, _) = self.identifier("expected a case binding, `_`, or type")?;
            let (binding, annotation) = if self.at(&TokenKind::Colon) {
                self.next();
                let binding = (first != "_").then_some(first);
                let annotation = if self.at(&TokenKind::Newline)
                    || self.at_identifier("return")
                    || self.at_identifier("assert")
                    || self.at_identifier("if")
                    || self.at_identifier("match")
                {
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
            let (body, end) = if self.at(&TokenKind::Newline) {
                self.indented_block("case")?
            } else {
                let statement = self.block_statement()?;
                let end = self.tokens[self.cursor.saturating_sub(1)].span.end;
                (vec![statement], end)
            };
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

    fn select_statement(&mut self) -> Result<Statement, Diagnostic> {
        let start = self.next().span;
        if !self.at_identifier("with") {
            return Err(self.error("expected `with` after `select`"));
        }
        self.next();
        if !self.at_identifier("limit") {
            return Err(self.error("expected `limit` after `select with`"));
        }
        self.next();
        self.expect(&TokenKind::Equal, "expected `=` after select limit")?;
        let limit = self.expression(0)?;
        self.expect(&TokenKind::Colon, "expected `:` after select declaration")?;
        self.expect(
            &TokenKind::Newline,
            "expected a newline after select header",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected indented select cases")?;
        self.separators();
        let mut cases = Vec::new();
        let mut error_body = Vec::new();
        let mut end = limit.span.end;
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            if !self.at_identifier("case") {
                return Err(self.error("expected `case` in select body"));
            }
            let case_start = self.next().span;
            let (binding, _) = self.identifier("expected a select case binding")?;
            if self.at_identifier("from") {
                self.next();
                let channel = self.expression(0)?;
                self.expect(&TokenKind::Colon, "expected `:` after select channel")?;
                let (body, case_end) = self.indented_block("select case")?;
                end = case_end;
                cases.push(SelectCase {
                    binding,
                    channel,
                    body,
                    span: Span::new(case_start.source, case_start.start, case_end),
                });
            } else {
                if binding != "error" {
                    return Err(self.error("expected `from` after select case binding"));
                }
                if !error_body.is_empty() {
                    return Err(self.error("a select may contain only one error case"));
                }
                self.expect(&TokenKind::Colon, "expected `:` after select error case")?;
                let (body, case_end) = self.indented_block("select error case")?;
                end = case_end;
                error_body = body;
            }
            self.separators();
        }
        self.expect(&TokenKind::Dedent, "expected end of select cases")?;
        if cases.is_empty() {
            return Err(Diagnostic::new(
                "E000112",
                "a select requires at least one channel case",
                Some(start),
            ));
        }
        Ok(Statement::Select {
            limit,
            cases,
            error_body,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn type_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<TypeDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, name_span) = self.identifier("expected a type name")?;
        let (type_parameters, mut constraints, _) = self.type_parameters()?;
        let definition = if self.take(&TokenKind::Equal).is_some() {
            Some(self.type_annotation()?)
        } else {
            None
        };
        constraints.extend(self.declaration_constraints()?);
        let end = definition
            .as_ref()
            .map_or(name_span.end, |definition| definition.span.end);
        Ok(TypeDeclaration {
            decorators,
            name,
            type_parameters,
            constraints,
            definition,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn union_declaration(&mut self) -> Result<TypeDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a union name")?;
        self.expect(&TokenKind::Colon, "expected `:` after union name")?;
        self.expect(&TokenKind::Newline, "expected a newline after union name")?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented union body")?;
        self.separators();
        let mut members = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            members.push(self.type_annotation()?);
            if !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error("expected a newline after union member"));
            }
            self.separators();
        }
        if members.is_empty() {
            return Err(Diagnostic::new(
                "E000112",
                "a union declaration requires at least one member",
                Some(start),
            ));
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of union body")?
            .span
            .end;
        Ok(TypeDeclaration {
            decorators: Vec::new(),
            name,
            type_parameters: Vec::new(),
            constraints: Vec::new(),
            definition: Some(TypeAnnotation {
                kind: TypeAnnotationKind::Union(members),
                span: Span::new(start.source, start.start, end),
            }),
            span: Span::new(start.source, start.start, end),
        })
    }

    fn import_declaration(&mut self) -> Result<ImportDeclaration, Diagnostic> {
        let keyword = self.next();
        let start = keyword.span;
        if matches!(&keyword.kind, TokenKind::Identifier(name) if name == "from") {
            let (source, _) = self.identifier("expected an import source after `from`")?;
            if !self.at_identifier("import") {
                return Err(self.error("expected `import` after import source"));
            }
            self.next();
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
            let alias = if self.at_identifier("as") {
                self.next();
                let (alias, span) = self.identifier("expected an import alias")?;
                end = span.end;
                Some(alias)
            } else {
                None
            };
            return Ok(ImportDeclaration {
                subject,
                source: Some(source),
                alias,
                span: Span::new(start.source, start.start, end),
            });
        }
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
        let (type_parameters, mut constraints, _) = self.type_parameters()?;
        constraints.extend(self.declaration_constraints()?);
        self.expect(&TokenKind::Colon, "expected `:` after trait name")?;
        let mut bases = Vec::new();
        if !self.at(&TokenKind::Newline) {
            loop {
                bases.push(self.type_annotation()?);
                if self.take(&TokenKind::Plus).is_none() {
                    break;
                }
            }
        }
        self.expect(
            &TokenKind::Newline,
            "expected a newline after trait header; base traits do not take a trailing `:`",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        if !self.at(&TokenKind::Indent) {
            return Ok(TraitDeclaration {
                decorators,
                namespaces: Vec::new(),
                name,
                type_parameters,
                constraints,
                bases,
                properties: Vec::new(),
                methods: Vec::new(),
                operators: Vec::new(),
                span: Span::new(start.source, start.start, self.peek().span.start),
            });
        }
        self.expect(&TokenKind::Indent, "expected an indented trait body")?;
        let mut properties = Vec::new();
        let mut methods = Vec::new();
        let mut operators = Vec::new();
        let mut namespaces = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let mut member_has_body = false;
            let member_decorators = if self.at(&TokenKind::At) {
                self.decorators()?
            } else {
                Vec::new()
            };
            if self.at_identifier("def") {
                let method = self.function_declaration(member_decorators)?;
                member_has_body = method.body.is_some();
                methods.push(method);
            } else if self.at_identifier("operator") {
                operators.push(self.operator_declaration(member_decorators)?);
            } else if !member_decorators.is_empty() {
                namespaces.extend(member_decorators);
            } else if self.at_identifier("property") {
                properties.push(self.property()?);
            } else if self.at_identifier("pass") {
                self.next();
            } else if self.looks_like_member_property() {
                properties.push(self.member_property()?);
            } else if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                bases.push(self.type_annotation()?);
            } else {
                return Err(self.error(
                    "expected a property, `def`, `operator`, composed trait, or `pass` in trait body",
                ));
            }
            if !member_has_body && !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error("expected a newline after trait member"));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of trait body")?
            .span;
        Ok(TraitDeclaration {
            decorators,
            namespaces,
            name,
            type_parameters,
            constraints,
            bases,
            properties,
            methods,
            operators,
            span: Span::new(start.source, start.start, end.end),
        })
    }

    fn class_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<ClassDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a class name")?;
        let (type_parameters, mut constraints, type_parameter_defaults) = self.type_parameters()?;
        constraints.extend(self.declaration_constraints()?);
        self.expect(&TokenKind::Colon, "expected `:` after class name")?;

        let mut traits = Vec::new();
        if !self.at(&TokenKind::Newline) {
            loop {
                traits.push(self.type_annotation()?);
                if self.take(&TokenKind::Plus).is_none()
                    && self.take(&TokenKind::Comma).is_none()
                {
                    break;
                }
            }
            constraints.extend(self.declaration_constraints()?);
        }
        let primitive = self.take(&TokenKind::Colon).is_some();
        self.expect(&TokenKind::Newline, "expected a newline after class header")?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented class body")?;

        let mut fields = Vec::new();
        let mut aliases = Vec::new();
        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        let mut operators = Vec::new();
        let mut tests = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let mut member_has_body = false;
            let member_decorators = if self.at(&TokenKind::At) {
                self.decorators()?
            } else {
                Vec::new()
            };
            if self.at_identifier("def") {
                let function = self.function_declaration(member_decorators)?;
                if function.body.is_none() {
                    return Err(Diagnostic::new(
                        "E000122",
                        "class methods require a body",
                        Some(function.span),
                    ));
                }
                member_has_body = true;
                if function.name == name {
                    constructors.push(function);
                } else {
                    methods.push(function);
                }
            } else if self.at_identifier("operator") {
                operators.push(self.operator_implementation(member_decorators)?);
                member_has_body = true;
            } else if self.at_identifier("test") {
                tests.push(self.test_declaration()?);
                member_has_body = true;
            } else if matches!(self.peek().kind, TokenKind::String(_)) {
                // Standalone block strings are declaration documentation.
                self.next();
            } else if !member_decorators.is_empty() {
                return Err(self.error("expected `def` or `operator` after class member decorator"));
            } else if self.at_identifier("trait") {
                self.next();
                traits.push(self.type_annotation()?);
            } else if self.at_identifier("self")
                && self
                    .tokens
                    .get(self.cursor + 1)
                    .is_some_and(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"))
            {
                self.next();
                self.next();
                aliases.push(self.type_annotation()?);
            } else if self.at_identifier("pass") {
                self.next();
            } else if self.looks_like_member_property() {
                fields.push(self.member_property()?);
            } else {
                return Err(
                    self.error("expected a field, method, constructor, or `pass` in class body")
                );
            }
            if !member_has_body && !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error("expected a newline after class member"));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of class body")?
            .span;
        Ok(ClassDeclaration {
            decorators,
            name,
            primitive,
            type_parameters,
            type_parameter_defaults,
            constraints,
            traits,
            aliases,
            fields,
            constructors,
            methods,
            operators,
            tests,
            span: Span::new(start.source, start.start, end.end),
        })
    }

    fn extension_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<ExtensionDeclaration, Diagnostic> {
        let start = self.next().span;
        let (type_parameters, mut constraints, _) = self.type_parameters()?;
        constraints.extend(self.declaration_constraints()?);
        let target = self.type_annotation()?;
        self.expect(&TokenKind::Colon, "expected `:` after extension target")?;
        self.expect(
            &TokenKind::Newline,
            "expected a newline after extension header",
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented extension body")?;

        let mut methods = Vec::new();
        let mut operators = Vec::new();
        self.separators();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let member_decorators = if self.at(&TokenKind::At) {
                self.decorators()?
            } else {
                Vec::new()
            };
            if self.at_identifier("def") {
                let method = self.function_declaration(member_decorators)?;
                if method.body.is_none() {
                    return Err(Diagnostic::new(
                        "E000122",
                        "extension methods require a body",
                        Some(method.span),
                    ));
                }
                methods.push(method);
            } else if self.at_identifier("operator") {
                operators.push(self.operator_implementation(member_decorators)?);
            } else if !member_decorators.is_empty() {
                return Err(
                    self.error("expected `def` or `operator` after extension member decorator")
                );
            } else if self.at_identifier("pass") {
                self.next();
            } else {
                return Err(self.error("expected a method, operator, or `pass` in extension body"));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of extension body")?
            .span;
        Ok(ExtensionDeclaration {
            decorators,
            type_parameters,
            constraints,
            target,
            methods,
            operators,
            span: Span::new(start.source, start.start, end.end),
        })
    }

    fn enum_declaration(&mut self) -> Result<EnumDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected an enum name")?;
        self.expect(&TokenKind::Colon, "expected `:` after enum name")?;
        self.expect(&TokenKind::Newline, "expected a newline after enum header")?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::Indent, "expected an indented enum body")?;
        self.separators();
        let mut variants = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let (variant_name, variant_start) = self.identifier("expected an enum variant")?;
            let mut fields = Vec::new();
            let mut end = variant_start.end;
            if self.take(&TokenKind::LeftParen).is_some() {
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        let field_start = self.peek().span;
                        let (field_name, _) = self.identifier("expected a payload field name")?;
                        self.expect(&TokenKind::Colon, "expected `:` after payload field")?;
                        let annotation = self.type_annotation()?;
                        end = annotation.span.end;
                        fields.push(PropertyDeclaration {
                            name: field_name,
                            annotation,
                            default: None,
                            constraints: Vec::new(),
                            span: Span::new(field_start.source, field_start.start, end),
                        });
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                end = self
                    .expect(&TokenKind::RightParen, "expected `)` after enum payload")?
                    .span
                    .end;
            }
            let mut accepted_values = Vec::new();
            if self.take(&TokenKind::LeftBrace).is_some() {
                if self.at(&TokenKind::RightBrace) {
                    return Err(self.error("an enum accepted-value set cannot be empty"));
                }
                loop {
                    let expression = self.expression(0)?;
                    let ExpressionKind::Literal(value) = expression.kind else {
                        return Err(Diagnostic::new(
                            "E000112",
                            "an enum accepted value must be a literal",
                            Some(expression.span),
                        ));
                    };
                    accepted_values.push(value);
                    if self.take(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
                end = self
                    .expect(
                        &TokenKind::RightBrace,
                        "expected `}` after enum accepted values",
                    )?
                    .span
                    .end;
            }
            let mut transitions = Vec::new();
            if self.take(&TokenKind::Arrow).is_some() {
                loop {
                    let (transition, transition_span) =
                        self.identifier("expected a transition variant")?;
                    end = transition_span.end;
                    transitions.push(transition);
                    if self.take(&TokenKind::Pipe).is_none() {
                        break;
                    }
                }
            }
            variants.push(EnumVariant {
                name: variant_name,
                fields,
                accepted_values,
                transitions,
                span: Span::new(variant_start.source, variant_start.start, end),
            });
            if !self.at(&TokenKind::Newline) && !self.at(&TokenKind::Dedent) {
                return Err(self.error("expected a newline after enum variant"));
            }
            self.separators();
        }
        let end = self
            .expect(&TokenKind::Dedent, "expected end of enum body")?
            .span
            .end;
        if variants.is_empty() {
            return Err(Diagnostic::new(
                "E000123",
                "an enum requires at least one variant",
                Some(Span::new(start.source, start.start, end)),
            ));
        }
        Ok(EnumDeclaration {
            name,
            variants,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn type_parameters(
        &mut self,
    ) -> Result<
        (
            Vec<String>,
            Vec<GenericConstraint>,
            Vec<Option<TypeAnnotation>>,
        ),
        Diagnostic,
    > {
        let mut type_parameters = Vec::new();
        let mut constraints = Vec::new();
        let mut defaults = Vec::new();
        if self.take(&TokenKind::LeftBracket).is_some() {
            loop {
                let variadic = self.take(&TokenKind::Star).is_some();
                let (parameter, parameter_span) =
                    self.identifier("expected a generic parameter")?;
                type_parameters.push(parameter.clone());
                if variadic {
                    constraints.push(GenericConstraint::VariadicPack {
                        parameter: parameter.clone(),
                        span: parameter_span,
                    });
                }
                if self.take(&TokenKind::Colon).is_some() {
                    loop {
                        let bound = self.type_annotation()?;
                        constraints.push(GenericConstraint::Parameter {
                            parameter: parameter.clone(),
                            span: Span::new(
                                parameter_span.source,
                                parameter_span.start,
                                bound.span.end,
                            ),
                            bound,
                        });
                        if self.take(&TokenKind::Plus).is_none() {
                            break;
                        }
                    }
                }
                defaults.push(if self.take(&TokenKind::Equal).is_some() {
                    Some(self.type_annotation()?)
                } else {
                    None
                });
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.expect(
                &TokenKind::RightBracket,
                "expected `]` after type parameters",
            )?;
        }
        Ok((type_parameters, constraints, defaults))
    }

    fn declaration_constraints(&mut self) -> Result<Vec<GenericConstraint>, Diagnostic> {
        if !self.at_identifier("with") {
            return Ok(Vec::new());
        }
        self.next();
        self.expect(
            &TokenKind::LeftBrace,
            "expected `{` after declaration `with`",
        )?;
        let multiline = self.take(&TokenKind::Newline).is_some();
        while self.take(&TokenKind::Newline).is_some() {}
        if multiline {
            self.expect(
                &TokenKind::Indent,
                "expected indented declaration constraints",
            )?;
            self.separators();
        }
        let mut constraints = Vec::new();
        while !self.at(&TokenKind::RightBrace)
            && !self.at(&TokenKind::Dedent)
            && !self.at(&TokenKind::Eof)
        {
            let parameter_constraint = matches!(self.peek().kind, TokenKind::Identifier(_))
                && self
                    .tokens
                    .get(self.cursor + 1)
                    .is_some_and(|token| token.kind == TokenKind::Colon);
            if parameter_constraint {
                let (parameter, parameter_span) =
                    self.identifier("expected constrained parameter")?;
                self.expect(&TokenKind::Colon, "expected `:` after parameter")?;
                let bound = self.type_annotation()?;
                constraints.push(GenericConstraint::Parameter {
                    parameter,
                    span: Span::new(parameter_span.source, parameter_span.start, bound.span.end),
                    bound,
                });
            } else {
                constraints.push(GenericConstraint::Predicate(self.expression(0)?));
            }
            if self.take(&TokenKind::Comma).is_none()
                && !self.at(&TokenKind::Newline)
                && !self.at(&TokenKind::Dedent)
                && !self.at(&TokenKind::RightBrace)
            {
                return Err(self.error("expected a comma or newline after constraint"));
            }
            self.separators();
        }
        if multiline {
            self.expect(
                &TokenKind::Dedent,
                "expected end of declaration constraints",
            )?;
        }
        self.expect(
            &TokenKind::RightBrace,
            "expected `}` after declaration constraints",
        )?;
        Ok(constraints)
    }

    fn function_contracts(
        &mut self,
        type_parameters: &[String],
    ) -> Result<(Vec<GenericConstraint>, Vec<FunctionContract>), Diagnostic> {
        if !self.at_identifier("with") {
            return Ok((Vec::new(), Vec::new()));
        }
        self.next();
        if !self.at(&TokenKind::LeftBrace) {
            return Ok((
                vec![GenericConstraint::Predicate(self.expression(0)?)],
                Vec::new(),
            ));
        }
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(&TokenKind::LeftBrace, "expected `{` after function `with`")?;
        let multiline = self.take(&TokenKind::Newline).is_some();
        while self.take(&TokenKind::Newline).is_some() {}
        if multiline {
            self.expect(&TokenKind::Indent, "expected indented function contracts")?;
            self.separators();
        }
        let mut contracts = Vec::new();
        let mut constraints = Vec::new();
        while !self.at(&TokenKind::RightBrace)
            && !self.at(&TokenKind::Dedent)
            && !self.at(&TokenKind::Eof)
        {
            let start = self.peek().span;
            let parameter_constraint = matches!(self.peek().kind, TokenKind::Identifier(_))
                && self
                    .tokens
                    .get(self.cursor + 1)
                    .is_some_and(|token| token.kind == TokenKind::Colon);
            if parameter_constraint {
                let (parameter, parameter_span) =
                    self.identifier("expected constrained parameter")?;
                self.expect(&TokenKind::Colon, "expected `:` after parameter")?;
                let bound = self.type_annotation()?;
                constraints.push(GenericConstraint::Parameter {
                    parameter,
                    span: Span::new(parameter_span.source, parameter_span.start, bound.span.end),
                    bound,
                });
                if self.take(&TokenKind::Comma).is_none()
                    && !self.at(&TokenKind::Newline)
                    && !self.at(&TokenKind::Dedent)
                    && !self.at(&TokenKind::RightBrace)
                {
                    return Err(self.error("expected a comma or newline after constraint"));
                }
                self.separators();
                continue;
            }
            let deferred = if self.at_identifier("defer") {
                self.next();
                true
            } else {
                false
            };
            let condition = self.expression(0)?;
            let failure = if self.take(&TokenKind::Arrow).is_some() {
                Some(self.expression(0)?)
            } else {
                None
            };
            if !deferred
                && failure.is_none()
                && type_parameters
                    .iter()
                    .any(|parameter| expression_mentions(&condition, parameter))
            {
                constraints.push(GenericConstraint::Predicate(condition));
            } else {
                let end = failure
                    .as_ref()
                    .map_or(condition.span.end, |value| value.span.end);
                contracts.push(FunctionContract {
                    condition,
                    deferred,
                    failure,
                    span: Span::new(start.source, start.start, end),
                });
            }
            if self.take(&TokenKind::Comma).is_none()
                && !self.at(&TokenKind::Newline)
                && !self.at(&TokenKind::Dedent)
                && !self.at(&TokenKind::RightBrace)
            {
                return Err(self.error("expected a comma or newline after function contract"));
            }
            self.separators();
        }
        if multiline {
            self.expect(&TokenKind::Dedent, "expected end of function contracts")?;
        }
        self.expect(
            &TokenKind::RightBrace,
            "expected `}` after function contracts",
        )?;
        Ok((constraints, contracts))
    }

    fn property(&mut self) -> Result<PropertyDeclaration, Diagnostic> {
        let start = self.next().span;
        self.member_property_after_start(start)
    }

    fn member_property(&mut self) -> Result<PropertyDeclaration, Diagnostic> {
        let start = self.peek().span;
        self.member_property_after_start(start)
    }

    fn member_property_after_start(
        &mut self,
        start: Span,
    ) -> Result<PropertyDeclaration, Diagnostic> {
        let (name, _) = self.identifier("expected a property name")?;
        self.expect(&TokenKind::Colon, "expected `:` after property name")?;
        let annotation = self.type_annotation()?;
        let default = if self.take(&TokenKind::Equal).is_some() {
            Some(self.expression(0)?)
        } else {
            None
        };
        let constraints = if self.take(&TokenKind::LeftBrace).is_some() {
            self.property_constraints()?
        } else {
            Vec::new()
        };
        let end = constraints.last().map_or_else(
            || {
                default
                    .as_ref()
                    .map_or(annotation.span.end, |value| value.span.end)
            },
            |constraint| constraint.span.end,
        );
        Ok(PropertyDeclaration {
            name,
            annotation,
            default,
            constraints,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn property_constraints(&mut self) -> Result<Vec<PropertyConstraint>, Diagnostic> {
        let multiline = self.take(&TokenKind::Newline).is_some();
        while self.take(&TokenKind::Newline).is_some() {}
        if multiline {
            self.expect(&TokenKind::Indent, "expected indented property constraints")?;
            self.separators();
        }
        let mut constraints = Vec::new();
        while !self.at(&TokenKind::RightBrace)
            && !self.at(&TokenKind::Dedent)
            && !self.at(&TokenKind::Eof)
        {
            let start = self.peek().span;
            let condition = self.expression(0)?;
            let mut continuation = false;
            if !self.at(&TokenKind::Arrow) && self.at(&TokenKind::Newline) {
                let mut lookahead = self.cursor;
                while self
                    .tokens
                    .get(lookahead)
                    .is_some_and(|token| token.kind == TokenKind::Newline)
                {
                    lookahead += 1;
                }
                continuation = self
                    .tokens
                    .get(lookahead)
                    .is_some_and(|token| token.kind == TokenKind::Indent)
                    && self
                        .tokens
                        .get(lookahead + 1)
                        .is_some_and(|token| token.kind == TokenKind::Arrow);
                if continuation {
                    while self.take(&TokenKind::Newline).is_some() {}
                    self.expect(&TokenKind::Indent, "expected indented constraint failure")?;
                }
            }
            let failure = if self.take(&TokenKind::Arrow).is_some() {
                Some(self.expression(0)?)
            } else {
                None
            };
            let end = failure
                .as_ref()
                .map_or(condition.span.end, |failure| failure.span.end);
            constraints.push(PropertyConstraint {
                condition,
                failure,
                span: Span::new(start.source, start.start, end),
            });
            self.take(&TokenKind::Comma);
            self.separators();
            if continuation {
                self.expect(&TokenKind::Dedent, "expected end of constraint failure")?;
                self.separators();
            }
        }
        if multiline {
            self.expect(&TokenKind::Dedent, "expected end of property constraints")?;
        }
        self.expect(
            &TokenKind::RightBrace,
            "expected `}` after property constraints",
        )?;
        Ok(constraints)
    }

    fn looks_like_member_property(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
            && self
                .tokens
                .get(self.cursor + 1)
                .is_some_and(|token| token.kind == TokenKind::Colon)
    }

    fn operator_declaration(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<OperatorDeclaration, Diagnostic> {
        let (start, operator, tag, type_parameters, mut constraints, parameters, close) =
            self.operator_head()?;
        let result = if self.take(&TokenKind::Arrow).is_some() {
            self.type_annotation()?
        } else {
            TypeAnnotation::named("unit", Vec::new(), close)
        };
        constraints.extend(self.declaration_constraints()?);
        Ok(OperatorDeclaration {
            decorators,
            operator,
            tag,
            type_parameters,
            constraints,
            parameters,
            span: Span::new(start.source, start.start, result.span.end),
            result,
        })
    }

    fn operator_implementation(
        &mut self,
        decorators: Vec<Decorator>,
    ) -> Result<OperatorImplementation, Diagnostic> {
        let (start, operator, tag, type_parameters, mut constraints, parameters, close) =
            self.operator_head()?;
        let mut expression_body = None;
        let result = if self.take(&TokenKind::Arrow).is_some() {
            if matches!(operator, OperatorSyntax::If | OperatorSyntax::Else) {
                let result_start = self.cursor;
                let annotation = self.type_annotation()?;
                if self.at(&TokenKind::Colon) {
                    annotation
                } else {
                    self.cursor = result_start;
                    let expression = self.expression(0)?;
                    expression_body = Some(expression);
                    TypeAnnotation::named("bool", Vec::new(), close)
                }
            } else {
                self.type_annotation()?
            }
        } else {
            TypeAnnotation::named("unit", Vec::new(), close)
        };
        let (additional_constraints, contracts) = self.function_contracts(&type_parameters)?;
        constraints.extend(additional_constraints);
        if let Some(expression) = expression_body {
            let end = expression.span.end;
            return Ok(OperatorImplementation {
                decorators,
                operator,
                tag,
                type_parameters,
                constraints,
                parameters,
                contracts,
                result,
                body: vec![Statement::Return {
                    value: Some(expression),
                    span: Span::new(start.source, start.start, end),
                }],
                span: Span::new(start.source, start.start, end),
            });
        }
        self.expect(&TokenKind::Colon, "expected `:` after operator signature")?;
        if operator == OperatorSyntax::Conversion {
            let end = self.opaque_indented_block("conversion operator")?;
            return Ok(OperatorImplementation {
                decorators,
                operator,
                tag,
                type_parameters,
                constraints,
                parameters,
                contracts,
                result,
                body: Vec::new(),
                span: Span::new(start.source, start.start, end),
            });
        }
        let (body, end) = self.indented_block("operator")?;
        Ok(OperatorImplementation {
            decorators,
            operator,
            tag,
            type_parameters,
            constraints,
            parameters,
            contracts,
            result,
            body,
            span: Span::new(start.source, start.start, end),
        })
    }

    fn operator_head(
        &mut self,
    ) -> Result<
        (
            Span,
            OperatorSyntax,
            Option<String>,
            Vec<String>,
            Vec<GenericConstraint>,
            Vec<OperatorParameter>,
            Span,
        ),
        Diagnostic,
    > {
        let start = self.next().span;
        let operator_token = self.next();
        let operator = if operator_token.kind == TokenKind::LeftBracket {
            self.expect(&TokenKind::RightBracket, "expected `]` after `operator [`")?;
            // Indexed assignment uses the same semantic operator identity as
            // indexed access. Its unit result and second value parameter carry
            // the mutation contract, just as compound-assignment operators use
            // their underlying source operator identity.
            self.take(&TokenKind::Equal);
            OperatorSyntax::Index
        } else {
            operator_syntax(&operator_token.kind).ok_or_else(|| {
                Diagnostic::new(
                    "E000117",
                    "expected an operator name",
                    Some(operator_token.span),
                )
            })?
        };
        let (mut type_parameters, mut constraints, _) = self.type_parameters()?;
        let tag = constraints.iter().find_map(|constraint| match constraint {
            GenericConstraint::Parameter {
                parameter, bound, ..
            } if bound.simple_name() == Some("Y") => Some(parameter.clone()),
            _ => None,
        });
        if let Some(tag) = &tag {
            type_parameters.retain(|parameter| parameter != tag);
            constraints.retain(|constraint| {
                !matches!(constraint,
                    GenericConstraint::Parameter { parameter, bound, .. }
                    if parameter == tag && bound.simple_name() == Some("Y"))
            });
        }
        self.expect(&TokenKind::LeftParen, "expected `(` after operator")?;
        let mut parameters = Vec::new();
        if !self.at(&TokenKind::RightParen) {
            loop {
                let (name, span) = self.identifier("expected an operator parameter")?;
                if !self.at(&TokenKind::Colon)
                    && (name == "self" || operator == OperatorSyntax::Conversion)
                {
                    if operator == OperatorSyntax::Conversion
                        && name != "self"
                        && !type_parameters.contains(&name)
                    {
                        type_parameters.push(name);
                    }
                    if self.take(&TokenKind::Comma).is_some() {
                        continue;
                    }
                    break;
                }
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
        let close = self
            .expect(&TokenKind::RightParen, "expected `)` after parameters")?
            .span;
        Ok((start, operator, tag, type_parameters, constraints, parameters, close))
    }

    fn opaque_indented_block(&mut self, owner: &str) -> Result<u32, Diagnostic> {
        self.expect(
            &TokenKind::Newline,
            &format!("expected a newline before {owner} body"),
        )?;
        while self.take(&TokenKind::Newline).is_some() {}
        self.expect(
            &TokenKind::Indent,
            &format!("expected an indented {owner} body"),
        )?;
        let mut depth = 1usize;
        let mut end = self.peek().span.start;
        while depth > 0 && !self.at(&TokenKind::Eof) {
            let token = self.next();
            end = token.span.end;
            match token.kind {
                TokenKind::Indent => depth += 1,
                TokenKind::Dedent => depth -= 1,
                _ => {}
            }
        }
        if depth != 0 {
            return Err(self.error(&format!("expected end of {owner} body")));
        }
        Ok(end)
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
                mutable: false,
                update: false,
                preserve_error: false,
            });
        }
        let (name, name_span) = self.identifier("expected a binding name")?;
        let compound = match self.peek().kind {
            TokenKind::PlusEqual => Some(BinaryOperator::Add),
            TokenKind::MinusEqual => Some(BinaryOperator::Subtract),
            TokenKind::StarEqual => Some(BinaryOperator::Multiply),
            TokenKind::SlashEqual => Some(BinaryOperator::Divide),
            TokenKind::FloorDivideEqual => Some(BinaryOperator::FloorDivide),
            TokenKind::PercentEqual => Some(BinaryOperator::Remainder),
            TokenKind::AmpersandEqual => Some(BinaryOperator::BitwiseAnd),
            TokenKind::PipeEqual => Some(BinaryOperator::Pipe),
            TokenKind::CaretEqual => Some(BinaryOperator::BitwiseXor),
            TokenKind::ShiftLeftEqual => Some(BinaryOperator::ShiftLeft),
            TokenKind::ShiftRightEqual => Some(BinaryOperator::ShiftRight),
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
                mutable: false,
                update: true,
                preserve_error: false,
            });
        }
        let inferred = self.take(&TokenKind::ColonEqual).is_some();
        let annotation = if !inferred && self.take(&TokenKind::Colon).is_some() {
            Some(self.type_annotation()?)
        } else {
            None
        };
        if !inferred && annotation.is_some() && self.at(&TokenKind::LeftBrace) {
            self.next();
            self.property_constraints()?;
        }
        if !inferred && (self.at(&TokenKind::Newline) || self.at(&TokenKind::Dedent)) {
            if let Some(annotation) = annotation.as_ref() {
                let value = match annotation.simple_name() {
                    Some("string") => Expression {
                        kind: ExpressionKind::Literal(severian_ast::Literal::String(String::new())),
                        span: annotation.span,
                    },
                    Some(
                        "float" | "f8e4m3fn" | "f8e5m2" | "f16" | "bf16" | "f32" | "f64" | "f80"
                        | "f128",
                    ) => Expression {
                        kind: ExpressionKind::Literal(severian_ast::Literal::Float("0.0".into())),
                        span: annotation.span,
                    },
                    Some("bool") => Expression {
                        kind: ExpressionKind::Literal(severian_ast::Literal::Boolean(false)),
                        span: annotation.span,
                    },
                    _ => Expression {
                        kind: ExpressionKind::Literal(severian_ast::Literal::Integer("0".into())),
                        span: annotation.span,
                    },
                };
                return Ok(Binding {
                    name,
                    annotation: Some(annotation.clone()),
                    span: Span::new(name_span.source, name_span.start, value.span.end),
                    value,
                    mutable: false,
                    update: false,
                    preserve_error: false,
                });
            }
        }
        let typed_mutable =
            !inferred && annotation.is_some() && self.take(&TokenKind::ColonEqual).is_some();
        let preserve_error =
            !inferred && !typed_mutable && self.take(&TokenKind::QuestionEqual).is_some();
        if !inferred && !typed_mutable && !preserve_error {
            self.expect(
                &TokenKind::Equal,
                "expected `=`, `?=`, or `:=` after binding name",
            )?;
        }
        let value = self.expression(0)?;
        Ok(Binding {
            name,
            annotation,
            span: Span::new(name_span.source, name_span.start, value.span.end),
            value,
            mutable: inferred || typed_mutable,
            update: false,
            preserve_error,
        })
    }

    fn loop_guards(&mut self) -> Result<Vec<LoopGuard>, Diagnostic> {
        self.expect(&TokenKind::LeftBrace, "expected `{` after loop `with`")?;
        let multiline = self.take(&TokenKind::Newline).is_some();
        while self.take(&TokenKind::Newline).is_some() {}
        if multiline {
            self.expect(&TokenKind::Indent, "expected indented loop guards")?;
            self.separators();
        }
        let mut guards = Vec::new();
        while !self.at(&TokenKind::RightBrace)
            && !self.at(&TokenKind::Dedent)
            && !self.at(&TokenKind::Eof)
        {
            let start = self.peek().span;
            if !self.at_identifier("defer") {
                return Err(self.error("expected `defer` before loop guard"));
            }
            self.next();
            let condition = self.expression(0)?;
            self.expect(&TokenKind::Arrow, "expected `->` after loop guard")?;
            let action_token = self.peek().clone();
            let action = if self.at_identifier("continue") {
                self.next();
                LoopGuardAction::Continue
            } else if self.at_identifier("break") {
                self.next();
                LoopGuardAction::Break
            } else {
                return Err(self.error("expected `continue` or `break` after loop guard"));
            };
            guards.push(LoopGuard {
                condition,
                action,
                span: Span::new(start.source, start.start, action_token.span.end),
            });
            if self.take(&TokenKind::Comma).is_none()
                && !self.at(&TokenKind::Newline)
                && !self.at(&TokenKind::Dedent)
                && !self.at(&TokenKind::RightBrace)
            {
                return Err(self.error("expected a comma or newline after loop guard"));
            }
            self.separators();
        }
        if multiline {
            self.expect(&TokenKind::Dedent, "expected end of loop guards")?;
        }
        self.expect(&TokenKind::RightBrace, "expected `}` after loop guards")?;
        Ok(guards)
    }

    fn statement(&mut self) -> Result<Statement, Diagnostic> {
        let field_assignment = matches!(self.peek().kind, TokenKind::Identifier(_))
            && self
                .tokens
                .get(self.cursor + 1)
                .is_some_and(|token| token.kind == TokenKind::Dot)
            && matches!(
                self.tokens.get(self.cursor + 2).map(|token| &token.kind),
                Some(TokenKind::Identifier(_))
            )
            && self.tokens.get(self.cursor + 3).is_some_and(|token| {
                matches!(
                    token.kind,
                    TokenKind::Equal
                        | TokenKind::PlusEqual
                        | TokenKind::MinusEqual
                        | TokenKind::StarEqual
                        | TokenKind::SlashEqual
                        | TokenKind::PercentEqual
                )
            });
        if field_assignment {
            let (object, object_span) = self.identifier("expected an assignment object")?;
            self.next();
            let (field, field_span) = self.identifier("expected an assigned field")?;
            let assignment = self.next();
            let right = self.expression(0)?;
            let object_expression = Expression {
                kind: ExpressionKind::Name(object),
                span: object_span,
            };
            let operator = match assignment.kind {
                TokenKind::Equal => None,
                TokenKind::PlusEqual => Some(BinaryOperator::Add),
                TokenKind::MinusEqual => Some(BinaryOperator::Subtract),
                TokenKind::StarEqual => Some(BinaryOperator::Multiply),
                TokenKind::SlashEqual => Some(BinaryOperator::Divide),
                TokenKind::PercentEqual => Some(BinaryOperator::Remainder),
                _ => unreachable!("field assignment was matched above"),
            };
            let value = match operator {
                None => right,
                Some(operator) => Expression {
                    span: Span::new(object_span.source, object_span.start, right.span.end),
                    kind: ExpressionKind::Binary {
                        operator,
                        left: Box::new(Expression {
                            span: Span::new(object_span.source, object_span.start, field_span.end),
                            kind: ExpressionKind::Member {
                                object: Box::new(object_expression.clone()),
                                name: field.clone(),
                            },
                        }),
                        right: Box::new(right),
                    },
                },
            };
            Ok(Statement::FieldAssignment {
                object: object_expression,
                field,
                span: Span::new(object_span.source, object_span.start, value.span.end),
                value,
            })
        } else if self.looks_like_destructuring_binding() {
            let start = self.peek().span;
            let mut names = Vec::new();
            loop {
                names.push(self.identifier("expected a binding name")?.0);
                if self.take(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            let mutable = self.take(&TokenKind::ColonEqual).is_some();
            if !mutable {
                self.expect(
                    &TokenKind::Equal,
                    "expected `=` or `:=` after binding pattern",
                )?;
            }
            let value = self.expression(0)?;
            Ok(Statement::Destructure {
                names,
                mutable,
                span: Span::new(start.source, start.start, value.span.end),
                value,
            })
        } else if self.looks_like_binding() {
            Ok(Statement::Binding(self.binding()?))
        } else {
            let first = self.expression(0)?;
            let assignment = match self.peek().kind {
                TokenKind::Equal => Some(None),
                TokenKind::PlusEqual => Some(Some(BinaryOperator::Add)),
                TokenKind::MinusEqual => Some(Some(BinaryOperator::Subtract)),
                TokenKind::StarEqual => Some(Some(BinaryOperator::Multiply)),
                TokenKind::SlashEqual => Some(Some(BinaryOperator::Divide)),
                TokenKind::PercentEqual => Some(Some(BinaryOperator::Remainder)),
                _ => None,
            };
            if let (ExpressionKind::Index { object, index }, Some(operator)) =
                (first.kind.clone(), assignment)
            {
                self.next();
                let right = self.expression(0)?;
                let value = match operator {
                    None => right,
                    Some(operator) => Expression {
                        span: Span::new(first.span.source, first.span.start, right.span.end),
                        kind: ExpressionKind::Binary {
                            operator,
                            left: Box::new(first.clone()),
                            right: Box::new(right),
                        },
                    },
                };
                Ok(Statement::IndexAssignment {
                    object: *object,
                    index: *index,
                    span: Span::new(first.span.source, first.span.start, value.span.end),
                    value,
                })
            } else if matches!(first.kind, ExpressionKind::Await { .. })
                && self.take(&TokenKind::Comma).is_some()
            {
                let start = first.span;
                let mut awaits = vec![first];
                loop {
                    let task = self.expression(0)?;
                    let end = task.span.end;
                    awaits.push(if matches!(task.kind, ExpressionKind::Await { .. }) {
                        task
                    } else {
                        Expression {
                            kind: ExpressionKind::Await {
                                expression: Box::new(task),
                            },
                            span: Span::new(start.source, start.start, end),
                        }
                    });
                    if self.take(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
                let end = awaits.last().expect("await list is nonempty").span.end;
                Ok(Statement::Expression(Expression {
                    kind: ExpressionKind::Tuple(awaits),
                    span: Span::new(start.source, start.start, end),
                }))
            } else {
                Ok(Statement::Expression(first))
            }
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
                            | TokenKind::QuestionEqual
                            | TokenKind::Equal
                            | TokenKind::PlusEqual
                            | TokenKind::MinusEqual
                            | TokenKind::StarEqual
                            | TokenKind::SlashEqual
                            | TokenKind::FloorDivideEqual
                            | TokenKind::PercentEqual
                            | TokenKind::AmpersandEqual
                            | TokenKind::PipeEqual
                            | TokenKind::CaretEqual
                            | TokenKind::ShiftLeftEqual
                            | TokenKind::ShiftRightEqual
                    )
                }))
    }

    fn looks_like_destructuring_binding(&self) -> bool {
        let mut cursor = self.cursor;
        let mut names = 0;
        loop {
            if !self
                .tokens
                .get(cursor)
                .is_some_and(|token| matches!(token.kind, TokenKind::Identifier(_)))
            {
                return false;
            }
            names += 1;
            cursor += 1;
            if !self
                .tokens
                .get(cursor)
                .is_some_and(|token| token.kind == TokenKind::Comma)
            {
                break;
            }
            cursor += 1;
        }
        names > 1
            && self
                .tokens
                .get(cursor)
                .is_some_and(|token| matches!(token.kind, TokenKind::Equal | TokenKind::ColonEqual))
    }

    fn looks_like_prefix_typed_binding(&self) -> bool {
        let mut trial = Parser {
            tokens: self.tokens,
            cursor: self.cursor,
            continuation_indents: self.continuation_indents,
            operators: self.operators.clone(),
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
        let mut comparison_tail: Option<Expression> = None;
        loop {
            const RANGE_PRECEDENCE: u8 = 4;
            const SYMBOL_PACK_PRECEDENCE: u8 = 8;
            const CAST_PRECEDENCE: u8 = 9;
            if self.at_identifier("as") {
                if CAST_PRECEDENCE < minimum_precedence {
                    break;
                }
                self.next();
                let target = self.type_annotation()?;
                let span = Span::new(expression.span.source, expression.span.start, target.span.end);
                expression = Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(Expression {
                            kind: ExpressionKind::TypeApplication {
                                callee: Box::new(Expression {
                                    kind: ExpressionKind::Name("__as__".into()),
                                    span,
                                }),
                                arguments: vec![target],
                            },
                            span,
                        }),
                        arguments: vec![CallArgument {
                            name: None,
                            spread: false,
                            value: expression,
                            expected_error: None,
                            span,
                        }],
                    },
                    span,
                };
                comparison_tail = None;
                continue;
            }
            if self.at(&TokenKind::Range) {
                if RANGE_PRECEDENCE < minimum_precedence {
                    break;
                }
                self.next();
                let right = self.expression(RANGE_PRECEDENCE + 1)?;
                let span = Span::new(
                    expression.span.source,
                    expression.span.start,
                    right.span.end,
                );
                expression = Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(Expression {
                            kind: ExpressionKind::Name("range".into()),
                            span,
                        }),
                        arguments: vec![
                            CallArgument {
                                name: None,
                                spread: false,
                                value: expression,
                                expected_error: None,
                                span,
                            },
                            CallArgument {
                                name: None,
                                spread: false,
                                value: right,
                                expected_error: None,
                                span,
                            },
                        ],
                    },
                    span,
                };
                comparison_tail = None;
                continue;
            }
            let symbol_pack_operator = match &self.peek().kind {
                TokenKind::Identifier(symbol)
                    if symbol
                        .chars()
                        .next()
                        .is_some_and(|character| character.is_ascii_uppercase()) =>
                {
                    Some(symbol.clone())
                }
                _ => None,
            };
            if let Some(symbol) = symbol_pack_operator {
                if SYMBOL_PACK_PRECEDENCE < minimum_precedence {
                    break;
                }
                let operator_span = self.next().span;
                let right = self.expression(SYMBOL_PACK_PRECEDENCE + 1)?;
                let span = Span::new(
                    expression.span.source,
                    expression.span.start,
                    right.span.end,
                );
                let callee = Expression {
                    kind: ExpressionKind::Member {
                        object: Box::new(Expression {
                            kind: ExpressionKind::Name("__operator__".into()),
                            span: operator_span,
                        }),
                        name: symbol,
                    },
                    span: operator_span,
                };
                expression = Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(callee),
                        arguments: vec![
                            CallArgument {
                                name: None,
                                spread: false,
                                span: expression.span,
                                expected_error: None,
                                value: expression,
                            },
                            CallArgument {
                                name: None,
                                spread: false,
                                span: right.span,
                                expected_error: None,
                                value: right,
                            },
                        ],
                    },
                    span,
                };
                comparison_tail = None;
                continue;
            }
            let negated_contains = self.at_identifier("not")
                && self.tokens.get(self.cursor + 1).is_some_and(
                    |token| matches!(&token.kind, TokenKind::Identifier(value) if value == "in"),
                );
            let Some(operator) = (if negated_contains {
                Some(BinaryOperator::Contains)
            } else {
                binary_operator(&self.peek().kind)
            }) else {
                break;
            };
            let precedence = self
                .operators
                .get(&operator)
                .map_or_else(|| precedence(operator), |metadata| metadata.precedence);
            if precedence < minimum_precedence {
                break;
            }
            self.next();
            if negated_contains {
                self.next();
            }
            let right_precedence = if self
                .operators
                .get(&operator)
                .is_some_and(|metadata| metadata.right_associative)
                || operator == BinaryOperator::Power
            {
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
            let comparison = is_comparison(operator);
            let binary_left = if comparison {
                comparison_tail
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| expression.clone())
            } else {
                expression.clone()
            };
            let combined = Expression {
                kind: ExpressionKind::Binary {
                    operator,
                    left: Box::new(binary_left),
                    right: Box::new(right.clone()),
                },
                span,
            };
            expression = if comparison_tail.is_some() && comparison {
                Expression {
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::And,
                        left: Box::new(expression),
                        right: Box::new(combined),
                    },
                    span,
                }
            } else if negated_contains {
                Expression {
                    kind: ExpressionKind::Unary {
                        operator: UnaryOperator::Not,
                        operand: Box::new(combined),
                    },
                    span,
                }
            } else {
                combined
            };
            comparison_tail = comparison.then_some(right);
        }
        if minimum_precedence == 0 && self.at_identifier("if") {
            self.next();
            let condition = self.expression(1)?;
            if !self.at_identifier("else") {
                return Err(Diagnostic::new(
                    "E000112",
                    "expected `else` after conditional expression condition",
                    Some(self.peek().span),
                ));
            }
            self.next();
            let fallback = self.expression(0)?;
            let span = Span::new(
                expression.span.source,
                expression.span.start,
                fallback.span.end,
            );
            expression = Expression {
                kind: ExpressionKind::Conditional {
                    value: Box::new(expression),
                    condition: Box::new(condition),
                    fallback: Box::new(fallback),
                },
                span,
            };
        } else if minimum_precedence == 0 && self.at_identifier("else") {
            self.next();
            let fallback = self.expression(0)?;
            let span = Span::new(
                expression.span.source,
                expression.span.start,
                fallback.span.end,
            );
            expression = Expression {
                kind: ExpressionKind::Fallback {
                    value: Box::new(expression),
                    fallback: Box::new(fallback),
                },
                span,
            };
        }
        Ok(expression)
    }

    fn unary(&mut self) -> Result<Expression, Diagnostic> {
        if self.at_identifier("throw") {
            let start = self.next().span;
            let error = self.expression(0)?;
            return Ok(Expression {
                span: Span::new(start.source, start.start, error.span.end),
                kind: ExpressionKind::Throw {
                    error: Box::new(error),
                },
            });
        }
        if self.at_identifier("async") {
            let start = self.next().span;
            let expression = self.postfix()?;
            let mut owner = TaskOwner::Inferred;
            let mut locked = false;
            if self.at_identifier("with") {
                self.next();
                loop {
                    let (modifier, _) = self.identifier("expected a task owner or `lock`")?;
                    match modifier.as_str() {
                        "self" => owner = TaskOwner::SelfScope,
                        "runtime" => owner = TaskOwner::Runtime,
                        "lock" => locked = true,
                        _ => return Err(self.error("unknown async task modifier")),
                    }
                    if !self.at_identifier("and") {
                        break;
                    }
                    self.next();
                }
            }
            return Ok(Expression {
                span: Span::new(start.source, start.start, expression.span.end),
                kind: ExpressionKind::Async {
                    expression: Box::new(expression),
                    owner,
                    locked,
                },
            });
        }
        if self.at_identifier("await") {
            let start = self.next().span;
            let expression = self.unary()?;
            return Ok(Expression {
                span: Span::new(start.source, start.start, expression.span.end),
                kind: ExpressionKind::Await {
                    expression: Box::new(expression),
                },
            });
        }
        if self.at_identifier("borrow") {
            let start = self.next().span;
            let operator = if self.at_identifier("mut") {
                self.next();
                UnaryOperator::BorrowMut
            } else {
                UnaryOperator::Borrow
            };
            let operand = self.unary()?;
            return Ok(Expression {
                span: Span::new(start.source, start.start, operand.span.end),
                kind: ExpressionKind::Unary {
                    operator,
                    operand: Box::new(operand),
                },
            });
        }
        let operator = match &self.peek().kind {
            TokenKind::Ampersand => Some(UnaryOperator::AddressOf),
            TokenKind::Plus => Some(UnaryOperator::Positive),
            TokenKind::Minus => Some(UnaryOperator::Negative),
            TokenKind::Bang => Some(UnaryOperator::Not),
            TokenKind::Identifier(value) if value == "not" => Some(UnaryOperator::Not),
            TokenKind::Identifier(value) if value == "copy" || value == "clone" => {
                Some(UnaryOperator::Copy)
            }
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
            if self.at(&TokenKind::Newline) {
                let mut lookahead = self.cursor;
                while self
                    .tokens
                    .get(lookahead)
                    .is_some_and(|token| token.kind == TokenKind::Newline)
                {
                    lookahead += 1;
                }
                let indented = self
                    .tokens
                    .get(lookahead)
                    .is_some_and(|token| token.kind == TokenKind::Indent);
                let member = self
                    .tokens
                    .get(lookahead + usize::from(indented))
                    .is_some_and(|token| token.kind == TokenKind::Dot);
                if member && (indented || self.continuation_indents > 0) {
                    while self.take(&TokenKind::Newline).is_some() {}
                    if indented {
                        self.next();
                        self.continuation_indents += 1;
                    }
                }
            }
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
            } else if self.at(&TokenKind::LeftBracket)
                && (self.type_application_follows()
                    || matches!(
                        &expression.kind,
                        ExpressionKind::Name(name) if matches!(name.as_str(), "channel" | "Channel")
                    ))
            {
                self.next();
                let mut arguments = Vec::new();
                if !self.at(&TokenKind::RightBracket) {
                    loop {
                        arguments.push(self.type_annotation()?);
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                let end = self
                    .expect(
                        &TokenKind::RightBracket,
                        "expected `]` after type arguments",
                    )?
                    .span
                    .end;
                let span = Span::new(expression.span.source, expression.span.start, end);
                expression = Expression {
                    kind: ExpressionKind::TypeApplication {
                        callee: Box::new(expression),
                        arguments,
                    },
                    span,
                };
            } else if let Some(open) = self.take(&TokenKind::LeftBracket) {
                let start = if self.at(&TokenKind::RightBracket) {
                    Some(Box::new(Expression {
                        kind: ExpressionKind::Literal(Literal::Integer("0".into())),
                        span: open.span,
                    }))
                } else if self.at(&TokenKind::Colon) {
                    None
                } else {
                    let first = self.expression(0)?;
                    if self.take(&TokenKind::Comma).is_some() {
                        let mut indices = vec![first];
                        while !self.at(&TokenKind::RightBracket) {
                            indices.push(self.expression(0)?);
                            if self.take(&TokenKind::Comma).is_none() {
                                break;
                            }
                        }
                        let span = Span::new(
                            indices[0].span.source,
                            indices[0].span.start,
                            indices
                                .last()
                                .expect("an index tuple is non-empty")
                                .span
                                .end,
                        );
                        Some(Box::new(Expression {
                            kind: ExpressionKind::Tuple(indices),
                            span,
                        }))
                    } else {
                        Some(Box::new(first))
                    }
                };
                if self.take(&TokenKind::Colon).is_some() {
                    expression = self.slice_postfix(expression, start, false)?;
                } else {
                    let index = start.ok_or_else(|| self.error("expected an index expression"))?;
                    let end = self
                        .expect(&TokenKind::RightBracket, "expected `]` after index")?
                        .span
                        .end;
                    let span = Span::new(expression.span.source, expression.span.start, end);
                    expression = Expression {
                        kind: ExpressionKind::Index {
                            object: Box::new(expression),
                            index,
                        },
                        span,
                    };
                }
            } else if self.at(&TokenKind::LeftParen) && self.interval_slice_follows() {
                self.next();
                let start = if self.at(&TokenKind::Colon) {
                    None
                } else {
                    Some(Box::new(self.expression(0)?))
                };
                self.expect(&TokenKind::Colon, "expected `:` in interval slice")?;
                expression = self.slice_postfix(expression, start, true)?;
            } else if self.take(&TokenKind::LeftParen).is_some() {
                let throws_call = matches!(
                    &expression.kind,
                    ExpressionKind::Name(name) if name == "throws"
                );
                let mock_call = matches!(
                    &expression.kind,
                    ExpressionKind::Name(name) if name == "mock"
                );
                if mock_call {
                    let start = expression.span;
                    let (cases, fallback, end) = self.mock_cases()?;
                    expression = Expression {
                        kind: ExpressionKind::Mock {
                            cases,
                            fallback: Box::new(fallback),
                        },
                        span: Span::new(start.source, start.start, end),
                    };
                    continue;
                }
                self.line_breaks();
                let mut arguments = Vec::new();
                if !self.at(&TokenKind::RightParen) {
                    loop {
                        let start = self.peek().span;
                        let spread = self.take(&TokenKind::Ellipsis).is_some();
                        let name = if !spread
                            && matches!(self.peek().kind, TokenKind::Identifier(_))
                            && self
                                .tokens
                                .get(self.cursor + 1)
                                .is_some_and(|token| token.kind == TokenKind::Equal)
                        {
                            let name = self.identifier("expected an argument name")?.0;
                            self.expect(&TokenKind::Equal, "expected `=` after argument name")?;
                            Some(name)
                        } else {
                            None
                        };
                        let value = self.expression(0)?;
                        if throws_call && arguments.is_empty() {
                            self.line_breaks();
                        }
                        let expected_error = if throws_call
                            && arguments.is_empty()
                            && self.take(&TokenKind::Arrow).is_some()
                        {
                            Some(self.expression(0)?)
                        } else {
                            None
                        };
                        let end = expected_error
                            .as_ref()
                            .map_or(value.span.end, |error| error.span.end);
                        arguments.push(CallArgument {
                            name,
                            spread,
                            span: Span::new(start.source, start.start, end),
                            value,
                            expected_error,
                        });
                        self.line_breaks();
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                        self.line_breaks();
                        if self.at(&TokenKind::RightParen) {
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

    fn mock_cases(&mut self) -> Result<(Vec<severian_ast::MockCase>, Expression, u32), Diagnostic> {
        self.line_breaks();
        let mut cases = Vec::new();
        let fallback = loop {
            if self.at_identifier("else") {
                self.next();
                let fallback = self.expression(0)?;
                self.line_breaks();
                break fallback;
            }
            if self.at(&TokenKind::RightParen) {
                let span = self.peek().span;
                let message = Expression {
                    kind: ExpressionKind::Literal(severian_ast::Literal::String(
                        "unmatched mock call".into(),
                    )),
                    span,
                };
                let error_name = Expression {
                    kind: ExpressionKind::Name("Error".into()),
                    span,
                };
                let error = Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(error_name),
                        arguments: vec![CallArgument {
                            name: None,
                            spread: false,
                            value: message,
                            expected_error: None,
                            span,
                        }],
                    },
                    span,
                };
                break Expression {
                    kind: ExpressionKind::Throw {
                        error: Box::new(error),
                    },
                    span,
                };
            }
            let call = self.expression(0)?;
            self.expect(&TokenKind::Arrow, "expected `->` after mocked call")?;
            let result = self.expression(0)?;
            cases.push(severian_ast::MockCase {
                span: Span::new(call.span.source, call.span.start, result.span.end),
                call,
                result,
            });
            self.line_breaks();
            self.take(&TokenKind::Comma);
            self.line_breaks();
        };
        let end = self
            .expect(
                &TokenKind::RightParen,
                "expected `)` after mock declaration",
            )?
            .span
            .end;
        if cases.is_empty() {
            return Err(self.error("a mock requires at least one call case"));
        }
        Ok((cases, fallback, end))
    }

    fn type_application_follows(&self) -> bool {
        let mut cursor = self.cursor + 1;
        let mut depth = 1usize;
        while let Some(token) = self.tokens.get(cursor) {
            match token.kind {
                TokenKind::LeftBracket => depth += 1,
                TokenKind::RightBracket => {
                    depth -= 1;
                    if depth == 0 {
                        return self
                            .tokens
                            .get(cursor + 1)
                            .is_some_and(|token| token.kind == TokenKind::LeftParen);
                    }
                }
                TokenKind::Colon if depth == 1 => return false,
                TokenKind::Float(_)
                | TokenKind::MeasuredNumber { .. }
                | TokenKind::String(_)
                    if depth == 1 =>
                {
                    return false;
                }
                _ => {}
            }
            cursor += 1;
        }
        false
    }

    fn interval_slice_follows(&self) -> bool {
        let mut cursor = self.cursor + 1;
        let mut nested = 0usize;
        while let Some(token) = self.tokens.get(cursor) {
            match token.kind {
                TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace => nested += 1,
                TokenKind::RightParen | TokenKind::RightBracket | TokenKind::Comma
                    if nested == 0 =>
                {
                    return false
                }
                TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace => {
                    nested = nested.saturating_sub(1)
                }
                TokenKind::Colon if nested == 0 => return true,
                _ => {}
            }
            cursor += 1;
        }
        false
    }

    fn slice_postfix(
        &mut self,
        object: Expression,
        start: Option<Box<Expression>>,
        start_exclusive: bool,
    ) -> Result<Expression, Diagnostic> {
        let at_close = |parser: &Self| {
            parser.at(&TokenKind::RightBracket) || parser.at(&TokenKind::RightParen)
        };
        let end = if self.at(&TokenKind::Colon) || at_close(self) {
            None
        } else {
            Some(Box::new(self.expression(0)?))
        };
        let step = if self.take(&TokenKind::Colon).is_some() {
            if at_close(self) {
                None
            } else {
                Some(Box::new(self.expression(0)?))
            }
        } else {
            None
        };
        let closing = self.next();
        let end_inclusive = start_exclusive && closing.kind == TokenKind::RightBracket;
        if !matches!(
            closing.kind,
            TokenKind::RightBracket | TokenKind::RightParen
        ) {
            return Err(Diagnostic::new(
                "E000112",
                "expected `]` or `)` after slice",
                Some(closing.span),
            ));
        }
        let span = Span::new(object.span.source, object.span.start, closing.span.end);
        Ok(Expression {
            kind: ExpressionKind::Slice {
                object: Box::new(object),
                start,
                end,
                step,
                start_exclusive,
                end_inclusive,
            },
            span,
        })
    }

    fn primary(&mut self) -> Result<Expression, Diagnostic> {
        let token = self.next();
        let kind = match token.kind {
            TokenKind::Integer(value) => ExpressionKind::Literal(Literal::Integer(value)),
            TokenKind::Float(value) => ExpressionKind::Literal(Literal::Float(value)),
            TokenKind::MeasuredNumber { magnitude, suffix } => {
                ExpressionKind::Literal(Literal::Measured { magnitude, suffix })
            }
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
            TokenKind::Colon => {
                let (symbol, symbol_span) =
                    self.identifier("expected a symbol name after `:`")?;
                return Ok(Expression {
                    kind: ExpressionKind::Symbol(symbol),
                    span: Span::new(token.span.source, token.span.start, symbol_span.end),
                });
            }
            TokenKind::Identifier(value) if value == "lambda" => {
                let mut parameters = Vec::new();
                if !self.at(&TokenKind::Colon) {
                    loop {
                        parameters.push(self.identifier("expected a lambda parameter")?.0);
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::Colon, "expected `:` after lambda parameters")?;
                let body = self.expression(0)?;
                let end = body.span.end;
                return Ok(Expression {
                    kind: ExpressionKind::Lambda {
                        parameters,
                        body: Box::new(body),
                    },
                    span: Span::new(token.span.source, token.span.start, end),
                });
            }
            TokenKind::Identifier(name) => ExpressionKind::Name(name),
            TokenKind::Pipe => {
                let mut parameters = Vec::new();
                if !self.at(&TokenKind::Pipe) {
                    loop {
                        parameters.push(self.identifier("expected a lambda parameter")?.0);
                        if self.take(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::Pipe, "expected `|` after lambda parameters")?;
                let body = self.expression(0)?;
                let end = body.span.end;
                return Ok(Expression {
                    kind: ExpressionKind::Lambda {
                        parameters,
                        body: Box::new(body),
                    },
                    span: Span::new(token.span.source, token.span.start, end),
                });
            }
            TokenKind::LeftBracket => {
                self.line_breaks();
                if self.at(&TokenKind::Star) {
                    let target = self.type_annotation()?;
                    self.expect(
                        &TokenKind::RightBracket,
                        "expected `]` after pointer cast type",
                    )?;
                    self.expect(
                        &TokenKind::LeftParen,
                        "expected `(` after pointer cast type",
                    )?;
                    let value = self.expression(0)?;
                    let end = self
                        .expect(
                            &TokenKind::RightParen,
                            "expected `)` after pointer cast value",
                        )?
                        .span
                        .end;
                    let value_span = value.span;
                    let callee = Expression {
                        kind: ExpressionKind::TypeApplication {
                            callee: Box::new(Expression {
                                kind: ExpressionKind::Name("__pointer_cast".into()),
                                span: token.span,
                            }),
                            arguments: vec![target],
                        },
                        span: Span::new(token.span.source, token.span.start, value_span.start),
                    };
                    return Ok(Expression {
                        kind: ExpressionKind::Call {
                            callee: Box::new(callee),
                            arguments: vec![severian_ast::CallArgument {
                                name: None,
                                spread: false,
                                value,
                                expected_error: None,
                                span: value_span,
                            }],
                        },
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                if self.at(&TokenKind::RightBracket) {
                    let end = self.next().span.end;
                    return Ok(Expression {
                        kind: ExpressionKind::List(Vec::new()),
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                let first = self.expression(0)?;
                if self.at_identifier("for") {
                    let clauses = self.comprehension_clauses()?;
                    let end = self
                        .expect(&TokenKind::RightBracket, "expected `]` after comprehension")?
                        .span
                        .end;
                    return Ok(Expression {
                        kind: ExpressionKind::ListComprehension {
                            value: Box::new(first),
                            clauses,
                        },
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                let mut values = vec![first];
                self.line_breaks();
                while self.take(&TokenKind::Comma).is_some() {
                    self.line_breaks();
                    if self.at(&TokenKind::RightBracket) {
                        break;
                    }
                    values.push(self.expression(0)?);
                    self.line_breaks();
                }
                let end = self
                    .expect(&TokenKind::RightBracket, "expected `]` after list literal")?
                    .span
                    .end;
                return Ok(Expression {
                    kind: ExpressionKind::List(values),
                    span: Span::new(token.span.source, token.span.start, end),
                });
            }
            TokenKind::LeftBrace => {
                self.line_breaks();
                if self.at(&TokenKind::RightBrace) {
                    let end = self.next().span.end;
                    return Ok(Expression {
                        kind: ExpressionKind::Map(Vec::new()),
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                let first = self.expression(0)?;
                if self.take(&TokenKind::Colon).is_some() {
                    let first_value = self.expression(0)?;
                    if self.at_identifier("for") {
                        let clauses = self.comprehension_clauses()?;
                        let end = self
                            .expect(&TokenKind::RightBrace, "expected `}` after comprehension")?
                            .span
                            .end;
                        return Ok(Expression {
                            kind: ExpressionKind::MapComprehension {
                                key: Box::new(first),
                                value: Box::new(first_value),
                                clauses,
                            },
                            span: Span::new(token.span.source, token.span.start, end),
                        });
                    }
                    let mut entries = vec![severian_ast::MapEntry {
                        span: Span::new(first.span.source, first.span.start, first_value.span.end),
                        key: first,
                        value: first_value,
                    }];
                    self.line_breaks();
                    while self.take(&TokenKind::Comma).is_some() {
                        self.line_breaks();
                        if self.at(&TokenKind::RightBrace) {
                            break;
                        }
                        let key = self.expression(0)?;
                        self.expect(&TokenKind::Colon, "expected `:` after map key")?;
                        let value = self.expression(0)?;
                        entries.push(severian_ast::MapEntry {
                            span: Span::new(key.span.source, key.span.start, value.span.end),
                            key,
                            value,
                        });
                        self.line_breaks();
                    }
                    let end = self
                        .expect(&TokenKind::RightBrace, "expected `}` after map literal")?
                        .span
                        .end;
                    return Ok(Expression {
                        kind: ExpressionKind::Map(entries),
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                if self.at_identifier("for") {
                    let clauses = self.comprehension_clauses()?;
                    let end = self
                        .expect(&TokenKind::RightBrace, "expected `}` after comprehension")?
                        .span
                        .end;
                    return Ok(Expression {
                        kind: ExpressionKind::SetComprehension {
                            value: Box::new(first),
                            clauses,
                        },
                        span: Span::new(token.span.source, token.span.start, end),
                    });
                }
                let mut values = vec![first];
                self.line_breaks();
                while self.take(&TokenKind::Comma).is_some() {
                    self.line_breaks();
                    if self.at(&TokenKind::RightBrace) {
                        break;
                    }
                    values.push(self.expression(0)?);
                    self.line_breaks();
                }
                let end = self
                    .expect(&TokenKind::RightBrace, "expected `}` after set literal")?
                    .span
                    .end;
                return Ok(Expression {
                    kind: ExpressionKind::Set(values),
                    span: Span::new(token.span.source, token.span.start, end),
                });
            }
            TokenKind::LeftParen => {
                self.line_breaks();
                if self.take(&TokenKind::RightParen).is_some() {
                    return Ok(Expression {
                        kind: ExpressionKind::Literal(Literal::Unit),
                        span: token.span,
                    });
                }
                let first = self.expression(0)?;
                self.line_breaks();
                if self.take(&TokenKind::Comma).is_none() {
                    self.expect(&TokenKind::RightParen, "expected `)`")?;
                    return Ok(first);
                }
                let mut values = vec![first];
                self.line_breaks();
                while !self.at(&TokenKind::RightParen) {
                    values.push(self.expression(0)?);
                    self.line_breaks();
                    if self.take(&TokenKind::Comma).is_none() {
                        break;
                    }
                    self.line_breaks();
                }
                let end = self
                    .expect(&TokenKind::RightParen, "expected `)` after tuple literal")?
                    .span
                    .end;
                return Ok(Expression {
                    kind: ExpressionKind::Tuple(values),
                    span: Span::new(token.span.source, token.span.start, end),
                });
            }
            unexpected => {
                let diagnostic = Diagnostic::new(
                    "E000111",
                    "expected a literal or binding name",
                    Some(token.span),
                )
                .with_label(token.span, "an expression must start here")
                .with_note(
                    "expressions may start with a literal, binding name, `(`, or list literal",
                );
                return Err(if unexpected == TokenKind::Dot {
                    diagnostic.with_help(
                        "float literals require a leading digit; write `0.5` instead of `.5`",
                    )
                } else {
                    diagnostic.with_help("check for a missing expression or an extra delimiter")
                });
            }
        };
        Ok(Expression {
            kind,
            span: token.span,
        })
    }

    fn comprehension_clauses(
        &mut self,
    ) -> Result<Vec<severian_ast::ComprehensionClause>, Diagnostic> {
        let mut clauses = Vec::new();
        while self.at_identifier("for") {
            let start = self.next().span;
            let mut bindings = vec![self.identifier("expected a comprehension binding")?.0];
            if self.take(&TokenKind::Comma).is_some() {
                bindings.push(
                    self.identifier("expected a comprehension binding after `,`")?
                        .0,
                );
            }
            if !self.at_identifier("in") {
                return Err(self.error("expected `in` after comprehension binding(s)"));
            }
            self.next();
            // `if` introduces the optional comprehension filter here, rather
            // than a conditional expression belonging to the iterable.
            let iterable = self.expression(1)?;
            let condition = if self.at_identifier("if") {
                self.next();
                Some(self.expression(1)?)
            } else {
                None
            };
            let end = condition
                .as_ref()
                .map_or(iterable.span.end, |condition| condition.span.end);
            clauses.push(severian_ast::ComprehensionClause {
                bindings,
                iterable,
                condition,
                span: Span::new(start.source, start.start, end),
            });
        }
        Ok(clauses)
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
        if let Some(start) = self.take(&TokenKind::Star) {
            self.expect(
                &TokenKind::LeftBracket,
                "expected `[` after `*` in pointer type",
            )?;
            let pointee = self.type_annotation()?;
            let end = self
                .expect(&TokenKind::RightBracket, "expected `]` after pointer type")?
                .span
                .end;
            return Ok(TypeAnnotation::named(
                "pointer",
                vec![pointee],
                Span::new(start.span.source, start.span.start, end),
            ));
        }
        if self.at_identifier("borrow") || self.at_identifier("move") {
            let start = self.next().span;
            if self.at_identifier("mut") {
                self.next();
            }
            let mut annotation = self.type_primary()?;
            annotation.span = Span::new(start.source, start.start, annotation.span.end);
            return Ok(annotation);
        }
        if let Some(open) = self.take(&TokenKind::LeftParen) {
            self.line_breaks();
            let mut elements = Vec::new();
            if !self.at(&TokenKind::RightParen) {
                loop {
                    elements.push(self.type_annotation()?);
                    self.line_breaks();
                    if self.take(&TokenKind::Comma).is_none() {
                        break;
                    }
                    self.line_breaks();
                }
            }
            let close = self.expect(&TokenKind::RightParen, "expected `)` after tuple type")?;
            if self.take(&TokenKind::Arrow).is_some() {
                let result = self.type_annotation()?;
                let end = result.span.end;
                return Ok(TypeAnnotation {
                    kind: TypeAnnotationKind::Function {
                        parameters: elements,
                        result: Box::new(result),
                    },
                    span: Span::new(open.span.source, open.span.start, end),
                });
            }
            return Ok(TypeAnnotation::named(
                "tuple",
                elements,
                Span::new(open.span.source, open.span.start, close.span.end),
            ));
        }
        let token = self.next();
        let (mut name, start) = match token.kind {
            TokenKind::Identifier(name) => (name, token.span),
            TokenKind::Integer(value) => {
                let normalized = value.replace('_', "");
                let Ok(value) = normalized.parse::<u64>() else {
                    return Err(Diagnostic::new(
                        "E000110",
                        "a tensor dimension constant must fit in an unsigned 64-bit integer",
                        Some(token.span),
                    ));
                };
                return Ok(TypeAnnotation {
                    kind: TypeAnnotationKind::DimensionConstant(value),
                    span: token.span,
                });
            }
            _ => {
                return Err(Diagnostic::new(
                    "E000110",
                    "expected a type",
                    Some(token.span),
                ))
            }
        };
        let mut name_end = start.end;
        while self.take(&TokenKind::Dot).is_some() {
            let (member, member_span) = self.identifier("expected a type name after `.`")?;
            name.push('.');
            name.push_str(&member);
            name_end = member_span.end;
        }
        let mut arguments = Vec::new();
        let mut end = name_end;
        if self.take(&TokenKind::LeftBracket).is_some() {
            if !self.at(&TokenKind::RightBracket) {
                loop {
                    arguments.push(self.type_argument_annotation()?);
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

    fn type_argument_annotation(&mut self) -> Result<TypeAnnotation, Diagnostic> {
        if let Some(star) = self.take(&TokenKind::Star) {
            let (name, name_span) = self.identifier("expected a shape parameter after `*`")?;
            return Ok(TypeAnnotation {
                kind: TypeAnnotationKind::ShapeSpread(name),
                span: Span::new(star.span.source, star.span.start, name_span.end),
            });
        }
        let mut annotation = self.type_annotation()?;
        while matches!(self.peek().kind, TokenKind::Slash | TokenKind::Star) {
            let operator = self.next();
            let right = self.type_annotation()?;
            let name = if operator.kind == TokenKind::Slash {
                "__dimension_divide"
            } else {
                "__dimension_multiply"
            };
            let span = Span::new(annotation.span.source, annotation.span.start, right.span.end);
            annotation = TypeAnnotation::named(name, vec![annotation, right], span);
        }
        Ok(annotation)
    }

    fn separators(&mut self) {
        while self.at(&TokenKind::Newline) || self.at(&TokenKind::Comma) {
            self.cursor += 1;
        }
    }

    fn statement_separators(&mut self) {
        self.separators();
        while self.continuation_indents > 0 && self.take(&TokenKind::Dedent).is_some() {
            self.continuation_indents -= 1;
            self.separators();
        }
    }

    fn line_breaks(&mut self) {
        while self.at(&TokenKind::Newline)
            || self.at(&TokenKind::Indent)
            || self.at(&TokenKind::Dedent)
        {
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
                let value = parse_interpolation(source, span)?;
                parts.push(Expression {
                    kind: ExpressionKind::Call {
                        callee: Box::new(Expression {
                            kind: ExpressionKind::Name("string".into()),
                            span,
                        }),
                        arguments: vec![CallArgument {
                            name: None,
                            spread: false,
                            value,
                            expected_error: None,
                            span,
                        }],
                    },
                    span,
                });
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
    let mut parser = Parser::new(&tokens);
    let expression = parser.expression(0)?;
    if parser.take(&TokenKind::Colon).is_some() {
        if parser.at(&TokenKind::Eof) {
            return Err(Diagnostic::new(
                "E000113",
                "expected a format specifier after `:`",
                Some(outer_span),
            ));
        }
        while !parser.at(&TokenKind::Eof) {
            parser.next();
        }
    }
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
        TokenKind::Operator(symbol) => OperatorSyntax::from_spelling(symbol),
        TokenKind::Identifier(value) if value == "if" => OperatorSyntax::If,
        TokenKind::Identifier(value) if value == "else" => OperatorSyntax::Else,
        TokenKind::Pipe => OperatorSyntax::Pipe,
        TokenKind::PipeEqual => OperatorSyntax::Pipe,
        TokenKind::Ampersand => OperatorSyntax::BitwiseAnd,
        TokenKind::AmpersandEqual => OperatorSyntax::BitwiseAnd,
        TokenKind::Caret => OperatorSyntax::BitwiseXor,
        TokenKind::CaretEqual => OperatorSyntax::BitwiseXor,
        TokenKind::Identifier(value) if value == "and" => OperatorSyntax::And,
        TokenKind::Identifier(value) if value == "or" => OperatorSyntax::Or,
        TokenKind::Identifier(value) if value == "not" => OperatorSyntax::Not,
        TokenKind::Plus => OperatorSyntax::Plus,
        TokenKind::PlusEqual => OperatorSyntax::Plus,
        TokenKind::Minus => OperatorSyntax::Minus,
        TokenKind::MinusEqual => OperatorSyntax::Minus,
        TokenKind::Star => OperatorSyntax::Multiply,
        TokenKind::StarEqual => OperatorSyntax::Multiply,
        TokenKind::Slash => OperatorSyntax::Divide,
        TokenKind::SlashEqual => OperatorSyntax::Divide,
        TokenKind::FloorDivide | TokenKind::FloorDivideEqual => OperatorSyntax::FloorDivide,
        TokenKind::Percent => OperatorSyntax::Remainder,
        TokenKind::PercentEqual => OperatorSyntax::Remainder,
        TokenKind::Power => OperatorSyntax::Power,
        TokenKind::ShiftLeft | TokenKind::ShiftLeftEqual => OperatorSyntax::ShiftLeft,
        TokenKind::ShiftRight | TokenKind::ShiftRightEqual => OperatorSyntax::ShiftRight,
        TokenKind::Conversion => OperatorSyntax::Conversion,
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

/// Collect parser mechanics from source operator-symbol declarations before
/// parsing expressions. This is a syntax prepass over the same token stream,
/// not a second frontend: the ordinary parser still produces the sole AST.
fn source_operator_table(tokens: &[Token]) -> BTreeMap<OperatorSyntax, ParserOperator> {
    let mut table = BTreeMap::new();
    for (class, token) in tokens.iter().enumerate() {
        if !matches!(&token.kind, TokenKind::Identifier(value) if value == "class") {
            continue;
        }
        let Some(body) = tokens[class..]
            .iter()
            .position(|token| token.kind == TokenKind::Indent)
            .map(|offset| class + offset + 1)
        else {
            continue;
        };
        let mut depth = 1usize;
        let mut end = body;
        while end < tokens.len() && depth > 0 {
            match tokens[end].kind {
                TokenKind::Indent => depth += 1,
                TokenKind::Dedent => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        let body = &tokens[body..end.saturating_sub(1)];
        let Some(symbol) = operator_string_property(body, "symbol") else {
            continue;
        };
        let Some(precedence) = operator_integer_property(body, "precedence")
            .and_then(|value| u8::try_from(value).ok())
        else {
            continue;
        };
        if operator_name_property(body, "fixity").as_deref() != Some("Infix") {
            continue;
        }
        let right_associative =
            operator_name_property(body, "associativity").as_deref() == Some("Right");
        table.insert(
            OperatorSyntax::from_spelling(&symbol),
            ParserOperator {
                precedence,
                right_associative,
            },
        );
    }
    table
}

fn operator_property_value<'a>(tokens: &'a [Token], property: &str) -> Option<&'a TokenKind> {
    let start = tokens.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(value) if value == property),
    )?;
    tokens[start + 1..]
        .iter()
        .take_while(|token| token.kind != TokenKind::Newline)
        .find_map(|token| match token.kind {
            TokenKind::String(_) | TokenKind::Integer(_) => Some(&token.kind),
            _ => None,
        })
}

fn operator_string_property(tokens: &[Token], property: &str) -> Option<String> {
    match operator_property_value(tokens, property)? {
        TokenKind::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn operator_integer_property(tokens: &[Token], property: &str) -> Option<u64> {
    match operator_property_value(tokens, property)? {
        TokenKind::Integer(value) => value.parse().ok(),
        _ => None,
    }
}

fn operator_name_property(tokens: &[Token], property: &str) -> Option<String> {
    let start = tokens.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(value) if value == property),
    )?;
    tokens[start + 1..]
        .iter()
        .take_while(|token| token.kind != TokenKind::Newline)
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(value) => Some(value.clone()),
            _ => None,
        })
        .last()
}

fn operator_spelling(operator: OperatorSyntax) -> &'static str {
    operator
        .standard_spelling()
        .unwrap_or("<source-defined operator>")
}

fn binary_operator(kind: &TokenKind) -> Option<BinaryOperator> {
    if matches!(kind, TokenKind::Identifier(value) if value == "is") {
        return Some(BinaryOperator::Identity);
    }
    let operator = operator_syntax(kind)?;
    (!matches!(operator, OperatorSyntax::Index | OperatorSyntax::If | OperatorSyntax::Else
        | OperatorSyntax::Conversion | OperatorSyntax::Not))
        .then_some(operator)
}

fn is_comparison(operator: BinaryOperator) -> bool {
    matches!(
        operator,
        BinaryOperator::Equal
            | BinaryOperator::Identity
            | BinaryOperator::NotEqual
            | BinaryOperator::Less
            | BinaryOperator::LessEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterEqual
    )
}

fn is_throws_call_statement(statement: &Statement) -> bool {
    matches!(
        statement,
        Statement::Expression(Expression {
            kind: ExpressionKind::Call { callee, .. },
            ..
        }) if matches!(callee.kind, ExpressionKind::Name(ref name) if name == "throws")
    )
}

fn expression_mentions(expression: &Expression, expected: &str) -> bool {
    match &expression.kind {
        ExpressionKind::Name(name) => name == expected,
        ExpressionKind::List(values)
        | ExpressionKind::Set(values)
        | ExpressionKind::Tuple(values) => values
            .iter()
            .any(|value| expression_mentions(value, expected)),
        ExpressionKind::Map(entries) => entries.iter().any(|entry| {
            expression_mentions(&entry.key, expected) || expression_mentions(&entry.value, expected)
        }),
        ExpressionKind::ListComprehension { value, clauses }
        | ExpressionKind::SetComprehension { value, clauses } => {
            expression_mentions(value, expected)
                || clauses.iter().any(|clause| {
                    expression_mentions(&clause.iterable, expected)
                        || clause
                            .condition
                            .as_ref()
                            .is_some_and(|condition| expression_mentions(condition, expected))
                })
        }
        ExpressionKind::MapComprehension {
            key,
            value,
            clauses,
        } => {
            expression_mentions(key, expected)
                || expression_mentions(value, expected)
                || clauses.iter().any(|clause| {
                    expression_mentions(&clause.iterable, expected)
                        || clause
                            .condition
                            .as_ref()
                            .is_some_and(|condition| expression_mentions(condition, expected))
                })
        }
        ExpressionKind::Mock { cases, fallback } => {
            cases.iter().any(|case| {
                expression_mentions(&case.call, expected)
                    || expression_mentions(&case.result, expected)
            }) || expression_mentions(fallback, expected)
        }
        ExpressionKind::Lambda { parameters, body } => {
            !parameters.iter().any(|parameter| parameter == expected)
                && expression_mentions(body, expected)
        }
        ExpressionKind::Member { object, .. } => expression_mentions(object, expected),
        ExpressionKind::Index { object, index } => {
            expression_mentions(object, expected) || expression_mentions(index, expected)
        }
        ExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            expression_mentions(object, expected)
                || [start, end, step]
                    .into_iter()
                    .flatten()
                    .any(|value| expression_mentions(value, expected))
        }
        ExpressionKind::TypeApplication { callee, .. } => expression_mentions(callee, expected),
        ExpressionKind::Call { callee, arguments } => {
            expression_mentions(callee, expected)
                || arguments
                    .iter()
                    .any(|argument| expression_mentions(&argument.value, expected))
        }
        ExpressionKind::Async { expression, .. } | ExpressionKind::Await { expression } => {
            expression_mentions(expression, expected)
        }
        ExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            expression_mentions(value, expected)
                || expression_mentions(condition, expected)
                || expression_mentions(fallback, expected)
        }
        ExpressionKind::Fallback { value, fallback } => {
            expression_mentions(value, expected) || expression_mentions(fallback, expected)
        }
        ExpressionKind::Throw { error } => expression_mentions(error, expected),
        ExpressionKind::Unary { operand, .. } => expression_mentions(operand, expected),
        ExpressionKind::Binary { left, right, .. } => {
            expression_mentions(left, expected) || expression_mentions(right, expected)
        }
        ExpressionKind::Literal(_) | ExpressionKind::Symbol(_) => false,
    }
}

fn precedence(operator: BinaryOperator) -> u8 {
    match operator {
        BinaryOperator::Or => 1,
        BinaryOperator::And => 2,
        BinaryOperator::Equal
        | BinaryOperator::Identity
        | BinaryOperator::NotEqual
        | BinaryOperator::Less
        | BinaryOperator::LessEqual
        | BinaryOperator::Greater
        | BinaryOperator::GreaterEqual
        | BinaryOperator::Contains => 3,
        BinaryOperator::Pipe => 4,
        BinaryOperator::BitwiseXor => 5,
        BinaryOperator::BitwiseAnd => 6,
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => 7,
        BinaryOperator::Add | BinaryOperator::Subtract => 7,
        BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::FloorDivide
        | BinaryOperator::Remainder => 8,
        BinaryOperator::Power => 9,
        _ => 7,
    }
}
