use crate::{Diagnostic, DiagnosticBag, SourceRange};
use severian_ast::{
    Block, ConstructorDecl, Expr, FunctionDecl, Ident, ImportKind, Item, Module, Pattern, Span,
    Stmt,
};
use severian_lexer::{Token, TokenKind};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

/// A source-level naming result. Renames are kept separate from diagnostics so
/// the command-line tool can apply only file-wide, collision-free changes.
#[derive(Debug, Default)]
pub struct NamingReport {
    pub diagnostics: DiagnosticBag,
    renames: BTreeMap<String, BTreeSet<String>>,
    direct_edits: Vec<TextEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TextEdit {
    span: Span,
    replacement: String,
}

#[derive(Debug, Clone, Copy)]
enum Role {
    Variable,
    Function,
    Type,
    Constant,
    Module,
    Decorator,
}

impl Role {
    fn code(self) -> &'static str {
        match self {
            Self::Variable => "N001",
            Self::Function => "N002",
            Self::Type => "N003",
            Self::Constant => "N004",
            Self::Module => "N005",
            Self::Decorator => "N006",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Variable => "variable must use snake_case",
            Self::Function => "function must use snake_case",
            Self::Type => "type must use PascalCase",
            Self::Constant => "constant must use UPPER_SNAKE_CASE",
            Self::Module => "module must use snake_case",
            Self::Decorator => "decorator must use snake_case",
        }
    }
}

pub fn check(module: &Module, tokens: &[Token], source: &str, path: &Path) -> NamingReport {
    let mut checker = Checker {
        source,
        path: path.to_path_buf(),
        report: NamingReport::default(),
    };
    checker.module(module);
    checker.compatibility_spellings(tokens);
    checker.report
}

/// Applies fixes only when every declaration with a given spelling agrees on
/// one replacement and that replacement does not already name another symbol.
pub fn apply_safe_fixes(source: &str, tokens: &[Token], report: &NamingReport) -> String {
    let identifiers = tokens
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut edits = report.direct_edits.clone();

    for (original, replacements) in &report.renames {
        let Some(replacement) = replacements.iter().next() else {
            continue;
        };
        if replacements.len() != 1
            || (replacement != original && identifiers.contains(replacement.as_str()))
        {
            continue;
        }
        edits.extend(tokens.iter().filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) if name == original => Some(TextEdit {
                span: token.span,
                replacement: replacement.clone(),
            }),
            _ => None,
        }));
    }

    edits.sort_by_key(|edit| (edit.span.start, edit.span.end));
    edits.dedup_by(|left, right| left.span == right.span && left.replacement == right.replacement);
    let mut output = source.to_owned();
    for edit in edits.into_iter().rev() {
        if edit.span.start <= edit.span.end
            && edit.span.end <= output.len()
            && output.is_char_boundary(edit.span.start)
            && output.is_char_boundary(edit.span.end)
        {
            output.replace_range(edit.span.start..edit.span.end, &edit.replacement);
        }
    }
    output
}

struct Checker<'source> {
    source: &'source str,
    path: PathBuf,
    report: NamingReport,
}

impl Checker<'_> {
    fn module(&mut self, module: &Module) {
        for item in &module.items {
            match item {
                Item::Function(function) => self.function(function, false),
                Item::Class(class) => {
                    self.name(&class.name, Role::Type);
                    for decorator in &class.decorators {
                        for segment in &decorator.name.segments {
                            self.name(segment, Role::Decorator);
                        }
                    }
                    for field in &class.fields {
                        self.name(&field.name, Role::Variable);
                        if let Some(value) = &field.default {
                            self.expression(value);
                        }
                    }
                    for constructor in &class.constructors {
                        self.constructor(constructor);
                    }
                    for method in &class.methods {
                        self.function(method, true);
                    }
                }
                Item::Trait(trait_) => {
                    self.name(&trait_.name, Role::Type);
                    for method in &trait_.methods {
                        self.name(&method.name, Role::Function);
                        for parameter in &method.params {
                            self.name(&parameter.name, Role::Variable);
                            if let Some(default) = &parameter.default {
                                self.expression(default);
                            }
                        }
                    }
                }
                Item::Enum(enum_) => {
                    self.name(&enum_.name, Role::Type);
                    for variant in &enum_.variants {
                        self.name(&variant.name, Role::Type);
                        for field in &variant.fields {
                            self.name(&field.name, Role::Variable);
                        }
                    }
                }
                Item::Import(import) => match &import.kind {
                    ImportKind::Module { path, alias } => {
                        for segment in path {
                            self.name(segment, Role::Module);
                        }
                        if let Some(alias) = alias {
                            self.name(alias, Role::Module);
                        }
                    }
                    ImportKind::From { module, names } => {
                        for segment in module {
                            self.name(segment, Role::Module);
                        }
                        for imported in names {
                            if let Some(alias) = &imported.alias {
                                self.name(alias, Role::Variable);
                            }
                        }
                    }
                },
                Item::Statement(Stmt::Let(binding)) => {
                    self.name(&binding.name, Role::Constant);
                    if let Some(value) = &binding.value {
                        self.expression(value);
                    }
                }
                Item::Statement(statement) => self.statement(statement),
            }
        }
    }

    fn function(&mut self, function: &FunctionDecl, method: bool) {
        if !(method && is_coordinate_accessor(&function.name.name)) {
            self.name(&function.name, Role::Function);
        }
        for decorator in &function.decorators {
            for segment in &decorator.name.segments {
                self.name(segment, Role::Decorator);
            }
        }
        for parameter in &function.params {
            self.name(&parameter.name, Role::Variable);
            if let Some(default) = &parameter.default {
                self.expression(default);
            }
        }
        self.block(&function.body);
        for test in &function.tests {
            self.block(&test.body);
        }
    }

    fn constructor(&mut self, constructor: &ConstructorDecl) {
        // Constructors deliberately carry their type's spelling.
        self.name(&constructor.name, Role::Type);
        for parameter in &constructor.params {
            self.name(&parameter.name, Role::Variable);
        }
        self.block(&constructor.body);
        for test in &constructor.tests {
            self.block(&test.body);
        }
    }

    fn block(&mut self, block: &Block) {
        for statement in &block.statements {
            self.statement(statement);
        }
    }

    fn statement(&mut self, statement: &Stmt) {
        match statement {
            Stmt::Let(binding) => {
                self.name(&binding.name, Role::Variable);
                if let Some(value) = &binding.value {
                    self.expression(value);
                }
            }
            Stmt::DestructureLet(binding) => {
                for name in &binding.names {
                    self.name(name, Role::Variable);
                }
                self.expression(&binding.value);
            }
            Stmt::TryBind(binding) => {
                self.name(&binding.name, Role::Variable);
                self.expression(&binding.value);
            }
            Stmt::Assign(statement) => {
                self.expression(&statement.target);
                self.expression(&statement.value);
            }
            Stmt::Assert(statement) => {
                self.expression(&statement.condition);
                if let Some(message) = &statement.message {
                    self.expression(message);
                }
            }
            Stmt::Return(statement) => {
                if let Some(value) = &statement.value {
                    self.expression(value);
                }
            }
            Stmt::If(statement) => {
                self.expression(&statement.condition);
                self.block(&statement.then_block);
                if let Some(branch) = &statement.else_branch {
                    match branch {
                        severian_ast::ElseBranch::If(branch) => {
                            self.statement(&Stmt::If((**branch).clone()))
                        }
                        severian_ast::ElseBranch::Block(block) => self.block(block),
                    }
                }
            }
            Stmt::While(statement) => {
                if let Some(setup) = &statement.setup {
                    self.statement(setup);
                }
                for capability in &statement.capabilities {
                    self.expression(capability);
                }
                self.expression(&statement.condition);
                self.block(&statement.body);
            }
            Stmt::For(statement) => {
                if let Some(setup) = &statement.setup {
                    self.statement(setup);
                }
                self.pattern(&statement.pattern);
                self.expression(&statement.iterable);
                self.block(&statement.body);
            }
            Stmt::Switch(statement) => {
                for value in &statement.values {
                    self.expression(value);
                }
                if let Some(condition) = &statement.repeat_condition {
                    self.expression(condition);
                }
                if let Some(setup) = &statement.setup {
                    self.statement(setup);
                }
                for arm in &statement.arms {
                    self.pattern(&arm.pattern);
                    if let Some(source) = &arm.source {
                        self.expression(source);
                    }
                    if let Some(guard) = &arm.guard {
                        self.expression(guard);
                    }
                    self.block(&arm.body);
                }
            }
            Stmt::With(statement) => {
                for resource in &statement.resources {
                    self.expression(resource);
                }
                self.block(&statement.body);
            }
            Stmt::Unsafe(statement) => self.block(&statement.body),
            Stmt::Expr(expression) => self.expression(expression),
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Identifier(name) => self.name(name, Role::Variable),
            Pattern::Tuple { elements, .. } | Pattern::List { elements, .. } => {
                for element in elements {
                    self.pattern(element);
                }
            }
            Pattern::Constructor { fields, .. } => {
                for field in fields {
                    self.pattern(field);
                }
            }
            Pattern::Or { alternatives, .. } => {
                for alternative in alternatives {
                    self.pattern(alternative);
                }
            }
            Pattern::Wildcard(_) | Pattern::Literal(_) => {}
        }
    }

    fn expression(&mut self, expression: &Expr) {
        match expression {
            Expr::Call(call) => {
                if let Expr::Member(member) = call.callee.as_ref() {
                    if !is_coordinate_accessor(&member.member.name) {
                        self.name_without_fix(&member.member, Role::Function);
                    }
                    self.expression(&member.object);
                } else {
                    self.expression(&call.callee);
                }
                for argument in &call.args {
                    if let Some(name) = &argument.name {
                        self.name_without_fix(name, Role::Variable);
                    }
                    self.expression(&argument.value);
                }
            }
            Expr::Member(member) => {
                self.expression(&member.object);
                self.name_without_fix(&member.member, Role::Variable);
            }
            Expr::Binary(binary) => {
                self.expression(&binary.left);
                self.expression(&binary.right);
            }
            Expr::Unary(unary) => self.expression(&unary.expr),
            Expr::List(collection) | Expr::Tuple(collection) | Expr::Set(collection) => {
                for element in &collection.elements {
                    self.expression(element);
                }
            }
            Expr::Map(map) => {
                for entry in &map.entries {
                    self.expression(&entry.key);
                    self.expression(&entry.value);
                }
            }
            Expr::Index(index) => {
                self.expression(&index.object);
                self.expression(&index.index);
            }
            Expr::Slice(slice) => {
                self.expression(&slice.object);
                for bound in [&slice.start, &slice.end, &slice.step]
                    .into_iter()
                    .flatten()
                {
                    self.expression(bound);
                }
            }
            Expr::If(if_) => {
                self.expression(&if_.condition);
                self.expression(&if_.then_expr);
                self.expression(&if_.else_expr);
            }
            Expr::Switch(switch) => {
                self.expression(&switch.value);
                for arm in &switch.arms {
                    self.pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.expression(guard);
                    }
                    self.expression(&arm.value);
                }
            }
            Expr::Lambda(lambda) => {
                for parameter in &lambda.params {
                    self.name(&parameter.name, Role::Variable);
                }
                match &lambda.body {
                    severian_ast::LambdaBody::Expr(expression) => self.expression(expression),
                    severian_ast::LambdaBody::Block(block) => self.block(block),
                }
            }
            Expr::ListComprehension(comprehension) => {
                self.expression(&comprehension.element);
                for clause in &comprehension.clauses {
                    self.pattern(&clause.pattern);
                    self.expression(&clause.iterable);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition);
                    }
                }
            }
            Expr::SetComprehension(comprehension) => {
                self.expression(&comprehension.element);
                for clause in &comprehension.clauses {
                    self.pattern(&clause.pattern);
                    self.expression(&clause.iterable);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition);
                    }
                }
            }
            Expr::MapComprehension(comprehension) => {
                self.expression(&comprehension.key);
                self.expression(&comprehension.value);
                for clause in &comprehension.clauses {
                    self.pattern(&clause.pattern);
                    self.expression(&clause.iterable);
                    if let Some(condition) = &clause.condition {
                        self.expression(condition);
                    }
                }
            }
            Expr::Await(value) => self.expression(&value.value),
            Expr::Async(value) => self.expression(&value.value),
            Expr::Channel(value) => self.expression(&value.capacity),
            Expr::Send(value) => {
                self.expression(&value.value);
                self.expression(&value.channel);
            }
            Expr::Ownership(value) => self.expression(&value.value),
            Expr::ChaosRule(value) => {
                self.expression(&value.function);
                self.expression(&value.value);
            }
            Expr::Identifier(_) | Expr::Literal(_) => {}
        }
    }

    fn name(&mut self, identifier: &Ident, role: Role) {
        self.check_name(identifier, role, true);
    }

    fn name_without_fix(&mut self, identifier: &Ident, role: Role) {
        self.check_name(identifier, role, false);
    }

    fn check_name(&mut self, identifier: &Ident, role: Role, fixable: bool) {
        let expected = expected_name(&identifier.name, role);
        if expected == identifier.name {
            return;
        }
        let code = if role_matches_canonical_operator(role, &identifier.name, &expected) {
            "N011"
        } else if role_matches_canonical_acronym(role, &identifier.name, &expected) {
            "N010"
        } else {
            role.code()
        };
        let message = if code == "N010" {
            format!(
                "`{}` must use the canonical technical spelling `{expected}`",
                identifier.name
            )
        } else if code == "N011" {
            format!(
                "`{}` must use the canonical scientific spelling `{expected}`",
                identifier.name
            )
        } else {
            format!(
                "{}: `{}` should be `{expected}`",
                role.description(),
                identifier.name
            )
        };
        let diagnostic = Diagnostic::warning(code, message)
            .with_help(format!("rename `{}` to `{expected}`", identifier.name))
            .at(self.range(identifier.span));
        self.report.diagnostics.push(diagnostic);
        if fixable {
            self.report
                .renames
                .entry(identifier.name.clone())
                .or_default()
                .insert(expected);
        }
    }

    fn compatibility_spellings(&mut self, tokens: &[Token]) {
        for (index, token) in tokens.iter().enumerate() {
            match &token.kind {
                TokenKind::Elif => {
                    self.deprecated(token.span, "`elif`", "`else <condition>:`", "else");
                }
                TokenKind::Else
                    if tokens
                        .get(index + 1)
                        .is_some_and(|next| next.kind == TokenKind::If) =>
                {
                    let next = &tokens[index + 1];
                    self.deprecated(
                        Span::new(token.span.start, next.span.end),
                        "`else if`",
                        "`else <condition>:`",
                        "else",
                    );
                }
                TokenKind::Identifier(name) if name == "impl" => {
                    self.deprecated(token.span, "`impl`", "`implement`", "implement");
                }
                _ => {}
            }
        }
    }

    fn deprecated(&mut self, span: Span, spelling: &str, preferred: &str, replacement: &str) {
        self.report.diagnostics.push(
            Diagnostic::warning(
                "N007",
                format!("{spelling} is a compatibility spelling; prefer {preferred}"),
            )
            .with_help(format!("replace {spelling} with `{replacement}`"))
            .at(self.range(span)),
        );
        self.report.direct_edits.push(TextEdit {
            span,
            replacement: replacement.to_owned(),
        });
    }

    fn range(&self, span: Span) -> SourceRange {
        let (start_line, start_column) = line_column(self.source, span.start);
        let (end_line, end_column) = line_column(self.source, span.end);
        SourceRange {
            file: self.path.clone(),
            start_byte: span.start,
            end_byte: span.end,
            start_line: Some(start_line),
            start_column: Some(start_column),
            end_line: Some(end_line),
            end_column: Some(end_column),
        }
    }
}

fn expected_name(name: &str, role: Role) -> String {
    match role {
        Role::Variable | Role::Function | Role::Module | Role::Decorator => to_snake_case(name),
        Role::Constant => to_snake_case(name).to_ascii_uppercase(),
        Role::Type => expected_type_name(name),
    }
}

fn expected_type_name(name: &str) -> String {
    if canonical_scientific(name).is_some_and(|canonical| canonical == name)
        || canonical_technical(name).is_some_and(|canonical| canonical == name)
        || is_short_generic(name)
    {
        return name.to_owned();
    }
    if let Some(canonical) = canonical_scientific(name).or_else(|| canonical_technical(name)) {
        return canonical.to_owned();
    }

    let words = words(name);
    let leading_acronyms = words
        .iter()
        .take_while(|word| is_known_acronym(word))
        .count();
    if leading_acronyms > 0 && name.starts_with(&words[0].to_ascii_uppercase()) {
        if leading_acronyms == 1 {
            return words.concat().to_ascii_lowercase();
        }
        return words
            .iter()
            .map(|word| word.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join("_");
    }

    words
        .iter()
        .map(|word| {
            let mut characters = word.chars();
            let Some(first) = characters.next() else {
                return String::new();
            };
            format!(
                "{}{}",
                first.to_ascii_uppercase(),
                characters.as_str().to_ascii_lowercase()
            )
        })
        .collect()
}

fn to_snake_case(name: &str) -> String {
    if name == "_" {
        return name.to_owned();
    }
    let prefix = if name.starts_with('_') { "_" } else { "" };
    let trimmed = name.trim_start_matches('_');
    let converted = words(trimmed)
        .iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("_");
    format!("{prefix}{converted}")
}

fn words(name: &str) -> Vec<String> {
    let mut output = Vec::new();
    for chunk in name.split('_').filter(|chunk| !chunk.is_empty()) {
        let mut index = 0;
        while index < chunk.len() {
            if let Some(acronym) = known_acronyms()
                .iter()
                .filter(|acronym| chunk[index..].starts_with(**acronym))
                .max_by_key(|acronym| acronym.len())
            {
                output.push((*acronym).to_owned());
                index += acronym.len();
                continue;
            }
            let start = index;
            let bytes = chunk.as_bytes();
            index += 1;
            if bytes[start].is_ascii_uppercase() {
                while index < bytes.len() && bytes[index].is_ascii_lowercase() {
                    index += 1;
                }
            } else {
                while index < bytes.len() && !bytes[index].is_ascii_uppercase() {
                    index += 1;
                }
            }
            output.push(chunk[start..index].to_owned());
        }
    }
    output
}

fn known_acronyms() -> &'static [&'static str] {
    &[
        "HTTPS",
        "StableHLO",
        "MLIR",
        "LLVM",
        "CUDA",
        "ROCm",
        "PJRT",
        "HTTP",
        "GPU",
        "CPU",
        "RPC",
        "XLA",
        "ABI",
        "AST",
        "HIR",
        "MIR",
        "JSON",
        "CSV",
        "TCP",
        "UDP",
        "URL",
        "URI",
        "AI",
        "ML",
    ]
}

fn is_known_acronym(value: &str) -> bool {
    known_acronyms()
        .iter()
        .any(|acronym| acronym.eq_ignore_ascii_case(value))
}

fn canonical_technical(value: &str) -> Option<&'static str> {
    [
        "BERT",
        "GPT",
        "CUDA",
        "ROCm",
        "MLIR",
        "XLA",
        "StableHLO",
        "PJRT",
    ]
    .into_iter()
    .find(|canonical| canonical.eq_ignore_ascii_case(value))
}

fn canonical_scientific(value: &str) -> Option<&'static str> {
    [
        "ReLU",
        "GELU",
        "SiLU",
        "LSTM",
        "GRU",
        "RMSNorm",
        "LayerNorm",
        "Softmax",
        "Conv2D",
    ]
    .into_iter()
    .find(|canonical| canonical.eq_ignore_ascii_case(value))
}

fn role_matches_canonical_operator(role: Role, original: &str, expected: &str) -> bool {
    matches!(role, Role::Type)
        && original != expected
        && canonical_scientific(original).is_some_and(|canonical| canonical == expected)
}

fn role_matches_canonical_acronym(role: Role, original: &str, expected: &str) -> bool {
    matches!(role, Role::Type)
        && original != expected
        && canonical_technical(original).is_some_and(|canonical| canonical == expected)
}

fn is_coordinate_accessor(name: &str) -> bool {
    matches!(name, "getX" | "getY" | "getZ" | "setX" | "setY" | "setZ")
}

fn is_short_generic(name: &str) -> bool {
    matches!(name, "T" | "K" | "V")
}

fn line_column(source: &str, byte: usize) -> (u32, u32) {
    let byte = byte.min(source.len());
    let line_start = source[..byte].rfind('\n').map_or(0, |index| index + 1);
    let line = source[..byte]
        .bytes()
        .filter(|value| *value == b'\n')
        .count() as u32
        + 1;
    let column = source[line_start..byte].chars().count() as u32 + 1;
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_lexer::lex;
    use severian_parser::parse;

    fn lint(source: &str) -> (NamingReport, Vec<Token>) {
        let tokens = lex(source).unwrap();
        let module = parse(&tokens).unwrap();
        (
            check(&module, &tokens, source, Path::new("test.sev")),
            tokens,
        )
    }

    #[test]
    fn fixes_role_based_names_and_references() {
        let source = concat!(
            "def LoadModel(modelPath: string):\n",
            "    hiddenState = modelPath\n",
            "    return hiddenState\n",
        );
        let (report, tokens) = lint(source);
        assert_eq!(report.diagnostics.warning_count(), 3);
        assert_eq!(
            apply_safe_fixes(source, &tokens, &report),
            concat!(
                "def load_model(model_path: string):\n",
                "    hidden_state = model_path\n",
                "    return hidden_state\n",
            )
        );
    }

    #[test]
    fn preserves_scientific_and_coordinate_spellings() {
        let source = concat!(
            "class ReLU:\n",
            "    value: int\n",
            "    def getX() -> int:\n",
            "        return value\n",
        );
        let (report, _) = lint(source);
        assert_eq!(report.diagnostics.warning_count(), 0);
    }

    #[test]
    fn tokenizes_adjacent_acronyms_explicitly() {
        assert_eq!(expected_type_name("HTTPServer"), "httpserver");
        assert_eq!(expected_type_name("HTTPRPCServer"), "http_rpc_server");
        assert_eq!(expected_type_name("XLAGPUExecutable"), "xla_gpu_executable");
        assert_eq!(expected_type_name("TransformerBlock"), "TransformerBlock");
    }

    #[test]
    fn fixes_python_elif_to_else_condition_syntax() {
        let source = concat!(
            "def classify(value: int) -> int:\n",
            "    if value > 0:\n",
            "        return 1\n",
            "    elif value < 0:\n",
            "        return -1\n",
            "    else:\n",
            "        return 0\n",
        );
        let (report, tokens) = lint(source);
        assert!(report
            .diagnostics
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code.0 == "N007"));
        let fixed = apply_safe_fixes(source, &tokens, &report);
        assert!(fixed.contains("else value < 0:"));
        parse(&lex(&fixed).unwrap()).unwrap();
    }
}
