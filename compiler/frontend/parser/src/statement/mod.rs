use severian_ast::{
    BinaryOperator, Binding, Expression, ExpressionKind, ImportDeclaration, Item, Literal, Module,
    OperatorDeclaration, OperatorParameter, OperatorSyntax, PropertyDeclaration, TraitDeclaration,
    TypeAnnotation, TypeAnnotationKind, UnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_lexer::{Token, TokenKind};
use severian_source::Span;

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
            if self.at_identifier("trait") {
                module.items.push(Item::Trait(self.trait_declaration()?));
                self.separators();
                continue;
            } else if self.at_identifier("import") {
                module.items.push(Item::Import(self.import_declaration()?));
            } else {
                module.items.push(Item::Binding(self.binding()?));
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

    fn import_declaration(&mut self) -> Result<ImportDeclaration, Diagnostic> {
        let start = self.next().span;
        let path_token = self.next();
        let TokenKind::String(path) = path_token.kind else {
            return Err(Diagnostic::new(
                "E000118",
                "expected an import path string",
                Some(path_token.span),
            ));
        };
        if !self.at_identifier("as") {
            return Err(self.error("expected `as` after import path"));
        }
        self.next();
        let (alias, end) = self.identifier("expected an import alias")?;
        Ok(ImportDeclaration {
            path,
            alias,
            span: Span::new(start.source, start.start, end.end),
        })
    }

    fn trait_declaration(&mut self) -> Result<TraitDeclaration, Diagnostic> {
        let start = self.next().span;
        let (name, _) = self.identifier("expected a trait name")?;
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
            } else {
                return Err(self.error("expected `property` or `operator` in trait body"));
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
            name,
            type_parameters,
            bases,
            properties,
            operators,
            span: Span::new(start.source, start.start, end.end),
        })
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
        let (name, name_span) = self.identifier("expected a binding name")?;
        let annotation = if self.take(&TokenKind::Colon).is_some() {
            Some(self.type_annotation()?)
        } else {
            None
        };
        self.expect(&TokenKind::Equal, "expected `=` after binding name")?;
        let value = self.expression(0)?;
        Ok(Binding {
            name,
            annotation,
            span: Span::new(name_span.source, name_span.start, value.span.end),
            value,
        })
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
        self.primary()
    }

    fn primary(&mut self) -> Result<Expression, Diagnostic> {
        let token = self.next();
        let kind = match token.kind {
            TokenKind::Integer(value) => ExpressionKind::Literal(Literal::Integer(value)),
            TokenKind::Float(value) => ExpressionKind::Literal(Literal::Float(value)),
            TokenKind::String(value) => ExpressionKind::Literal(Literal::String(value)),
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
        BinaryOperator::Add | BinaryOperator::Subtract => 4,
        BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Remainder => 5,
        BinaryOperator::Power => 6,
    }
}
