use super::model::{Mutation, MutationEdit, MutationKind};
use severian_ast::{
    BinaryOperator, ComprehensionClause, Expression, ExpressionKind, FunctionContract,
    FunctionDeclaration, GenericConstraint, Item, Literal, PropertyDeclaration, Statement,
};
use severian_lexer::{Token, TokenKind};
use severian_modules::{ModuleGraph, ResolvedModule};
use severian_source::Span;

pub(crate) fn discover(graph: &ModuleGraph) -> Result<Vec<Mutation>, String> {
    let mut mutations = Vec::new();
    let root_package = graph
        .modules
        .last()
        .map(|module| module.package)
        .expect("a resolved module graph contains its root");
    for module in graph
        .modules
        .iter()
        .filter(|module| module.package == root_package)
    {
        let tokens = severian_lexer::scan(&module.source)
            .map_err(|error| format!("could not scan {}: {error}", module.path.display()))?;
        let mut collector = Collector {
            module,
            tokens: &tokens,
            mutations: &mut mutations,
        };
        collector.module();
    }
    mutations.sort_by(|left, right| {
        left.file
            .cmp(&right.file)
            .then_with(|| left.span.start.cmp(&right.span.start))
            .then_with(|| left.span.end.cmp(&right.span.end))
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.replacement.cmp(&right.replacement))
    });
    for (index, mutation) in mutations.iter_mut().enumerate() {
        mutation.id = index + 1;
    }
    Ok(mutations)
}

struct Collector<'a> {
    module: &'a ResolvedModule,
    tokens: &'a [Token],
    mutations: &'a mut Vec<Mutation>,
}

impl Collector<'_> {
    fn module(&mut self) {
        for item in &self.module.ast.items {
            match item {
                Item::Import(_) | Item::Test(_) => {}
                Item::Binding(binding) => self.expression(&binding.value),
                Item::Expression(expression) => self.expression(expression),
                Item::Function(function) => self.function(function),
                Item::Type(declaration) => self.constraints(&declaration.constraints),
                Item::Trait(declaration) => {
                    self.constraints(&declaration.constraints);
                    for property in &declaration.properties {
                        self.property(property);
                    }
                    for method in &declaration.methods {
                        self.function(method);
                    }
                }
                Item::Class(declaration) => {
                    self.constraints(&declaration.constraints);
                    for field in &declaration.fields {
                        self.property(field);
                    }
                    for function in declaration.constructors.iter().chain(&declaration.methods) {
                        self.function(function);
                    }
                    for operator in &declaration.operators {
                        self.contracts(&operator.contracts);
                        self.statements(&operator.body);
                    }
                }
                Item::Extension(declaration) => {
                    for function in &declaration.methods {
                        self.function(function);
                    }
                    for operator in &declaration.operators {
                        self.contracts(&operator.contracts);
                        self.statements(&operator.body);
                    }
                }
                Item::Enum(declaration) => {
                    for variant in &declaration.variants {
                        for field in &variant.fields {
                            self.property(field);
                        }
                    }
                }
            }
        }
    }

    fn function(&mut self, function: &FunctionDeclaration) {
        self.constraints(&function.constraints);
        self.contracts(&function.contracts);
        for parameter in &function.parameters {
            if let Some(default) = &parameter.default {
                self.expression(default);
            }
        }
        if let Some(hook) = &function.hook {
            self.statements(&hook.with_phase);
            self.statements(&hook.without_phase);
        }
        if let Some(body) = &function.body {
            self.statements(body);
        }
    }

    fn property(&mut self, property: &PropertyDeclaration) {
        if let Some(default) = &property.default {
            self.expression(default);
        }
        for constraint in &property.constraints {
            self.expression(&constraint.condition);
            if let Some(failure) = &constraint.failure {
                self.expression(failure);
            }
        }
    }

    fn constraints(&mut self, constraints: &[GenericConstraint]) {
        for constraint in constraints {
            if let GenericConstraint::Predicate(expression) = constraint {
                self.expression(expression);
            }
        }
    }

    fn contracts(&mut self, contracts: &[FunctionContract]) {
        for contract in contracts {
            self.expression(&contract.condition);
            if let Some(failure) = &contract.failure {
                self.expression(failure);
            }
        }
    }

    fn statements(&mut self, statements: &[Statement]) {
        for statement in statements {
            self.statement(statement);
        }
    }

    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Binding(binding) => self.expression(&binding.value),
            Statement::Destructure { value, .. } => self.expression(value),
            Statement::FieldAssignment { object, value, .. } => {
                self.expression(object);
                self.expression(value);
            }
            Statement::IndexAssignment {
                object,
                index,
                value,
                ..
            } => {
                self.expression(object);
                self.expression(index);
                self.expression(value);
            }
            Statement::Expression(expression) | Statement::Defer { expression, .. } => {
                self.expression(expression);
            }
            Statement::Return { value, .. } => {
                if let Some(value) = value {
                    self.expression(value);
                }
            }
            Statement::Assert {
                condition, message, ..
            } => {
                self.expression(condition);
                if let Some(message) = message {
                    self.expression(message);
                }
            }
            Statement::Unsafe { body, .. } => self.statements(body),
            Statement::Try {
                body, catch_body, ..
            } => {
                self.statements(body);
                self.statements(catch_body);
            }
            Statement::FallibleElse { value, body, .. } => {
                self.expression(value);
                self.statements(body);
            }
            Statement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                self.push(
                    MutationKind::Conditional,
                    condition.span,
                    "condition",
                    "!condition",
                    MutationEdit::NegateConditional {
                        condition: condition.span,
                    },
                );
                self.expression(condition);
                self.statements(then_block);
                self.statements(else_block);
            }
            Statement::While {
                condition,
                initializer,
                guards,
                body,
                ..
            } => {
                self.expression(condition);
                if let Some(initializer) = initializer {
                    self.expression(&initializer.value);
                }
                for guard in guards {
                    self.expression(&guard.condition);
                }
                self.statements(body);
            }
            Statement::For {
                iterable,
                initializer,
                body,
                ..
            } => {
                self.expression(iterable);
                if let Some(initializer) = initializer {
                    self.expression(&initializer.value);
                }
                self.statements(body);
            }
            Statement::Match { subject, cases, .. } => {
                self.expression(subject);
                for case in cases {
                    self.statements(&case.body);
                }
            }
            Statement::Select {
                limit,
                cases,
                error_body,
                ..
            } => {
                self.expression(limit);
                for case in cases {
                    self.expression(&case.channel);
                    self.statements(&case.body);
                }
                self.statements(error_body);
            }
            Statement::Break { .. } | Statement::Continue { .. } => {}
        }
    }

    fn expression(&mut self, expression: &Expression) {
        match &expression.kind {
            ExpressionKind::Literal(Literal::Boolean(value)) => self.push(
                MutationKind::BooleanLiteral,
                expression.span,
                if *value { "true" } else { "false" },
                if *value { "false" } else { "true" },
                MutationEdit::Boolean {
                    expression: expression.span,
                    value: *value,
                },
            ),
            ExpressionKind::Literal(_) | ExpressionKind::Name(_) => {}
            ExpressionKind::List(values)
            | ExpressionKind::Set(values)
            | ExpressionKind::Tuple(values) => {
                for value in values {
                    self.expression(value);
                }
            }
            ExpressionKind::Map(entries) => {
                for entry in entries {
                    self.expression(&entry.key);
                    self.expression(&entry.value);
                }
            }
            ExpressionKind::ListComprehension { value, clauses }
            | ExpressionKind::SetComprehension { value, clauses } => {
                self.expression(value);
                self.clauses(clauses);
            }
            ExpressionKind::MapComprehension {
                key,
                value,
                clauses,
            } => {
                self.expression(key);
                self.expression(value);
                self.clauses(clauses);
            }
            ExpressionKind::Mock { cases, fallback } => {
                for case in cases {
                    self.expression(&case.call);
                    self.expression(&case.result);
                }
                self.expression(fallback);
            }
            ExpressionKind::Lambda { body, .. }
            | ExpressionKind::Async {
                expression: body, ..
            }
            | ExpressionKind::Await { expression: body }
            | ExpressionKind::Throw { error: body }
            | ExpressionKind::Unary { operand: body, .. } => self.expression(body),
            ExpressionKind::Member { object, .. } => self.expression(object),
            ExpressionKind::Index { object, index } => {
                self.expression(object);
                self.expression(index);
            }
            ExpressionKind::Slice {
                object,
                start,
                end,
                step,
                ..
            } => {
                self.expression(object);
                for value in [start, end, step].into_iter().flatten() {
                    self.expression(value);
                }
            }
            ExpressionKind::TypeApplication { callee, .. } => self.expression(callee),
            ExpressionKind::Call { callee, arguments } => {
                self.expression(callee);
                for argument in arguments {
                    self.expression(&argument.value);
                    if let Some(error) = &argument.expected_error {
                        self.expression(error);
                    }
                }
            }
            ExpressionKind::Conditional {
                value,
                condition,
                fallback,
            } => {
                self.expression(value);
                self.expression(condition);
                self.expression(fallback);
            }
            ExpressionKind::Fallback { value, fallback } => {
                self.expression(value);
                self.expression(fallback);
            }
            ExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                if let Some((kind, replacement)) = replacement(*operator) {
                    let original_text = operator_text(*operator);
                    if let Some(span) = self.operator_span(left, right, *operator) {
                        self.push(
                            kind,
                            span,
                            original_text,
                            operator_text(replacement),
                            MutationEdit::Binary {
                                expression: expression.span,
                                original: *operator,
                                replacement,
                            },
                        );
                    }
                }
                self.expression(left);
                self.expression(right);
            }
        }
    }

    fn clauses(&mut self, clauses: &[ComprehensionClause]) {
        for clause in clauses {
            self.expression(&clause.iterable);
            if let Some(condition) = &clause.condition {
                self.expression(condition);
            }
        }
    }

    fn operator_span(
        &self,
        left: &Expression,
        right: &Expression,
        operator: BinaryOperator,
    ) -> Option<Span> {
        self.tokens
            .iter()
            .find(|token| {
                token.span.start >= left.span.end
                    && token.span.end <= right.span.start
                    && token_matches(&token.kind, operator)
            })
            .map(|token| token.span)
    }

    fn push(
        &mut self,
        kind: MutationKind,
        span: Span,
        original: &str,
        replacement: &str,
        edit: MutationEdit,
    ) {
        if span.source != self.module.source.id
            || span.end < span.start
            || usize::try_from(span.end)
                .ok()
                .is_none_or(|end| end > self.module.source.text.len())
        {
            return;
        }
        self.mutations.push(Mutation {
            id: 0,
            kind,
            file: self.module.path.clone(),
            span,
            original: original.into(),
            replacement: replacement.into(),
            edit,
        });
    }
}

fn replacement(operator: BinaryOperator) -> Option<(MutationKind, BinaryOperator)> {
    Some(match operator {
        BinaryOperator::Equal => (MutationKind::Comparison, BinaryOperator::NotEqual),
        BinaryOperator::NotEqual => (MutationKind::Comparison, BinaryOperator::Equal),
        BinaryOperator::Less => (MutationKind::Comparison, BinaryOperator::LessEqual),
        BinaryOperator::LessEqual => (MutationKind::Comparison, BinaryOperator::Less),
        BinaryOperator::Greater => (MutationKind::Comparison, BinaryOperator::GreaterEqual),
        BinaryOperator::GreaterEqual => (MutationKind::Comparison, BinaryOperator::Greater),
        BinaryOperator::Add => (MutationKind::Arithmetic, BinaryOperator::Subtract),
        BinaryOperator::Subtract => (MutationKind::Arithmetic, BinaryOperator::Add),
        BinaryOperator::Multiply => (MutationKind::Arithmetic, BinaryOperator::Divide),
        BinaryOperator::Divide => (MutationKind::Arithmetic, BinaryOperator::Multiply),
        BinaryOperator::And => (MutationKind::Logical, BinaryOperator::Or),
        BinaryOperator::Or => (MutationKind::Logical, BinaryOperator::And),
        _ => return None,
    })
}

fn operator_text(operator: BinaryOperator) -> &'static str {
    match operator {
        BinaryOperator::Add => "+",
        BinaryOperator::Subtract => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Equal => "==",
        BinaryOperator::NotEqual => "!=",
        BinaryOperator::Less => "<",
        BinaryOperator::LessEqual => "<=",
        BinaryOperator::Greater => ">",
        BinaryOperator::GreaterEqual => ">=",
        BinaryOperator::And => "and",
        BinaryOperator::Or => "or",
        _ => unreachable!("only mutable operators are formatted"),
    }
}

fn token_matches(token: &TokenKind, operator: BinaryOperator) -> bool {
    matches!(
        (token, operator),
        (TokenKind::Plus, BinaryOperator::Add)
            | (TokenKind::Minus, BinaryOperator::Subtract)
            | (TokenKind::Star, BinaryOperator::Multiply)
            | (TokenKind::Slash, BinaryOperator::Divide)
            | (TokenKind::EqualEqual, BinaryOperator::Equal)
            | (TokenKind::NotEqual, BinaryOperator::NotEqual)
            | (TokenKind::Less, BinaryOperator::Less)
            | (TokenKind::LessEqual, BinaryOperator::LessEqual)
            | (TokenKind::Greater, BinaryOperator::Greater)
            | (TokenKind::GreaterEqual, BinaryOperator::GreaterEqual)
    ) || matches!(
        (token, operator),
        (TokenKind::Identifier(name), BinaryOperator::And) if name == "and"
    ) || matches!(
        (token, operator),
        (TokenKind::Identifier(name), BinaryOperator::Or) if name == "or"
    )
}
