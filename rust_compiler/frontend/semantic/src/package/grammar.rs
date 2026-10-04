use severian_ast::{self as ast, Expression as Ex, ExpressionKind as K, Item, Literal, Statement};
use severian_diagnostics::Diagnostic;
use severian_modules::{ModuleGraph, ResolvedImport, ResolvedModule};
use severian_source::Span;

// Both declarations own grammars. A trait is never converted into a fabricated
// class: its identity, property contracts, and receiver requirements survive.
struct GrammarOwner<'a> {
    name: &'a str,
    kind: &'static str,
    span: Span,
    type_parameters: &'a [String],
    type_parameter_defaults: &'a [Option<ast::TypeAnnotation>],
    constraints: &'a [ast::GenericConstraint],
    fields: &'a [ast::PropertyDeclaration],
    traits: &'a [ast::TypeAnnotation],
    sentences: &'a [ast::SentenceDeclaration],
}

impl<'a> GrammarOwner<'a> {
    fn from_item(item: &'a Item) -> Option<Self> {
        match item {
            Item::Class(owner) => Some(Self {
                name: &owner.name, kind: "class", span: owner.span,
                type_parameters: &owner.type_parameters,
                type_parameter_defaults: &owner.type_parameter_defaults,
                constraints: &owner.constraints, fields: &owner.fields,
                traits: &owner.traits, sentences: &owner.sentences,
            }),
            Item::Trait(owner) => Some(Self {
                name: &owner.name, kind: "trait", span: owner.span,
                type_parameters: &owner.type_parameters,
                type_parameter_defaults: &owner.type_parameter_defaults,
                constraints: &owner.constraints, fields: &owner.properties,
                traits: &owner.bases, sentences: &owner.sentences,
            }),
            _ => None,
        }
    }

    fn is_trait(&self) -> bool { self.kind == "trait" }
}

fn expression(kind: K, span: Span) -> Ex { Ex { kind, span } }
fn name(value: &str, span: Span) -> Ex { expression(K::Name(value.into()), span) }
fn text(value: &str, span: Span) -> Ex { expression(K::Literal(Literal::String(value.into())), span) }
fn call(callee: &str, values: Vec<Ex>, span: Span) -> Ex {
    expression(K::Call {
        callee: Box::new(name(callee, span)),
        arguments: values.into_iter().map(|value| ast::CallArgument {
            name: None, spread: false, value, expected_error: None, span,
        }).collect(),
    }, span)
}
fn annotation(value: &str, span: Span) -> ast::TypeAnnotation {
    ast::TypeAnnotation::named(value, Vec::new(), span)
}
fn import(module: &mut ResolvedModule, target: &ResolvedModule, alias: &str, _origin: Span) {
    // Import planning keys selections by the consumer's source offset. Each
    // synthesized edge needs its own key even when owners share source offsets.
    let mut position = 0;
    while module.imports.iter().any(|edge| edge.span.start == position) { position += 1; }
    let span = Span::new(module.source.id, position, position);
    module.ast.items.insert(0, Item::Import(ast::ImportDeclaration {
        subject: ast::ImportSubject::Locator(target.path.to_string_lossy().into_owned()),
        source: None, alias: Some(alias.into()), span,
    }));
    module.imports.push(ResolvedImport { span, module: target.id });
}

fn number(value: impl ToString, span: Span) -> Ex {
    expression(K::Literal(Literal::Integer(value.to_string())), span)
}

fn boolean(value: bool, span: Span) -> Ex {
    expression(K::Literal(Literal::Boolean(value)), span)
}

fn list(values: impl IntoIterator<Item = Ex>, span: Span) -> Ex {
    expression(K::List(values.into_iter().collect()), span)
}

fn source_span(span: Span) -> Ex {
    call("Span", vec![number(span.source, span), number(span.start, span), number(span.end, span)], span)
}

fn declaration_source(module: &ResolvedModule, span: Span) -> Ex {
    // Rust frontend spans are UTF-8 byte offsets; exported source spans use
    // Unicode scalar coordinates, as required by the source contract.
    let source = &module.source.text;
    let start = span.start as usize;
    let end = span.end as usize;
    let spelling = source.get(start..end).expect("grammar source span must belong to its original source snapshot");
    let scalar_span = Span::new(span.source,
        source.get(..start).expect("grammar start must be a UTF-8 boundary").chars().count() as u32,
        source.get(..end).expect("grammar end must be a UTF-8 boundary").chars().count() as u32);
    call("DeclarationSource", vec![text(spelling, span), source_span(scalar_span)], span)
}

fn type_contract(module: &ResolvedModule, value: &ast::TypeAnnotation) -> Ex {
    let span = value.span;
    let kind = match &value.kind {
        ast::TypeAnnotationKind::Named { name: spelling, arguments } => call("TypeAnnotationKind.Named", vec![text(spelling, span), list(arguments.iter().map(|item| type_contract(module, item)), span)], span),
        ast::TypeAnnotationKind::Union(members) => call("TypeAnnotationKind.Union", vec![list(members.iter().map(|item| type_contract(module, item)), span)], span),
        ast::TypeAnnotationKind::Function { parameters, result } => call("TypeAnnotationKind.Function", vec![list(parameters.iter().map(|item| type_contract(module, item)), span), type_contract(module, result)], span),
        ast::TypeAnnotationKind::DimensionConstant(value) => call("TypeAnnotationKind.DimensionConstant", vec![number(value, span)], span),
        ast::TypeAnnotationKind::DimensionRuntime(value) => call("TypeAnnotationKind.DimensionRuntime", vec![number(value, span)], span),
        ast::TypeAnnotationKind::ShapeSpread(value) => call("TypeAnnotationKind.ShapeSpread", vec![text(value, span)], span),
    };
    // Binding identifies the lexical module; names and generic arguments are
    // preserved, never resolved in the registry module's import environment.
    call("TypeAnnotation", vec![kind, source_span(Span::new(span.source,
        module.source.text.get(..span.start as usize).expect("type start must be a UTF-8 boundary").chars().count() as u32,
        module.source.text.get(..span.end as usize).expect("type end must be a UTF-8 boundary").chars().count() as u32)),
        text(&format!("{:032x}:{:032x}", module.package.0, module.id.0), span)], span)
}

fn constraint_contract(module: &ResolvedModule, constraint: &ast::GenericConstraint) -> Ex {
    match constraint {
        ast::GenericConstraint::Parameter { parameter, bound, span } => call("DeclaredConstraint.Parameter", vec![text(parameter, *span), type_contract(module, bound), declaration_source(module, *span)], *span),
        ast::GenericConstraint::VariadicPack { parameter, span } => call("DeclaredConstraint.VariadicPack", vec![text(parameter, *span), declaration_source(module, *span)], *span),
        ast::GenericConstraint::Predicate(value) => call("DeclaredConstraint.Predicate", vec![declaration_source(module, value.span)], value.span),
    }
}

fn grammar_contract(module: &ResolvedModule, owner: &GrammarOwner<'_>, grammar: &ast::SentenceDeclaration, ordinal: usize) -> Result<Ex, Diagnostic> {
    let span = grammar.function.span;
    let mut seen = vec![false; grammar.function.parameters.len()];
    for field in &grammar.fields {
        if let ast::SentenceElement::Capture(index) = field {
            let Some(used) = seen.get_mut(*index) else {
                return Err(Diagnostic::new("E000212", "grammar capture index is outside its callable parameters", Some(span)));
            };
            if *used { return Err(Diagnostic::new("E000212", "grammar capture occurs more than once; use a repeated capture", Some(span))); }
            *used = true;
        }
    }
    if seen.contains(&false) {
        return Err(Diagnostic::new("E000212", "grammar callable parameter has no capture", Some(span)));
    }
    let identity = |declaration: u128| call("DefId", vec![number(module.package.0, span), number(module.id.0, span), call("DeclarationId", vec![number(declaration, span)], span)], span);
    let owner_id = identity(super::stable_hash(&format!("{}:{}", owner.kind, owner.name)));
    let callable_id = identity(super::stable_hash(&format!("grammar:{}:{}:{ordinal}", owner.name, grammar.function.name)));
    let optional_source = |value: &Option<Ex>| value.as_ref().map(|value| declaration_source(module, value.span)).unwrap_or_else(|| expression(K::Literal(Literal::None), span));
    let captures = grammar.function.parameters.iter().map(|parameter| call("CaptureContract", vec![
        text(&parameter.name, parameter.span), type_contract(module, &parameter.annotation), boolean(parameter.variadic, parameter.span), boolean(parameter.immutable_reference, parameter.span), declaration_source(module, parameter.span), optional_source(&parameter.default),
    ], parameter.span));
    let fields = grammar.fields.iter().map(|field| match field {
        ast::SentenceElement::Literal(value) => call("DeclaredGrammarField.Literal", vec![text(value, span)], span),
        ast::SentenceElement::Capture(index) => call("DeclaredGrammarField.Capture", vec![number(index, span)], span),
    });
    let receiver = owner.fields.iter().map(|field| call("CaptureContract", vec![text(&field.name, field.span), type_contract(module, &field.annotation), boolean(false, field.span), boolean(false, field.span), declaration_source(module, field.span), optional_source(&field.default)], field.span));
    Ok(call("GrammarDeclarationContract", vec![
        call("GrammarCallable", vec![callable_id, owner_id, declaration_source(module, span)], span), declaration_source(module, owner.span), boolean(grammar.lexical, span), list(fields, span), list(captures, span), type_contract(module, &grammar.function.result),
        list(owner.type_parameters.iter().map(|value| text(value, span)), span), list(grammar.function.type_parameters.iter().map(|value| text(value, span)), span),
        list(owner.type_parameter_defaults.iter().map(|value| value.as_ref().map(|value| type_contract(module, value)).unwrap_or_else(|| expression(K::Literal(Literal::None), span))), span),
        list(owner.constraints.iter().map(|value| constraint_contract(module, value)), span), list(grammar.function.constraints.iter().map(|value| constraint_contract(module, value)), span), list(receiver, span), list(owner.traits.iter().map(|value| type_contract(module, value)), span),
    ], span))
}

// Executable adapters are a separate capability of the old parser. Failure to
// provide one must never remove a declaration from the registry.
fn supports_legacy_adapter(owner: &GrammarOwner<'_>, grammar: &ast::SentenceDeclaration) -> bool {
    if !owner.type_parameters.is_empty() || !grammar.function.type_parameters.is_empty()
        || !owner.constraints.is_empty()
        || grammar.function.constraints.iter().any(|item| !matches!(item, ast::GenericConstraint::Predicate(_))) {
        return false;
    }
    if grammar.lexical {
        return grammar.function.parameters.len() <= 1 && grammar.function.parameters.iter().all(|parameter| !parameter.variadic && parameter.annotation.simple_name() == Some("Lexeme"));
    }
    grammar.function.body.is_some() && grammar.function.parameters.iter().all(|parameter| parameter.annotation.simple_name().is_some())
}

fn attach_contract(entry: &mut Ex, contract: Ex) {
    let K::Tuple(pair) = &mut entry.kind else { unreachable!("registry entry is a pair") };
    let K::Call { arguments, .. } = &mut pair[1].kind else { unreachable!("registry descriptor is a constructor") };
    let span = contract.span;
    arguments.push(ast::CallArgument { name: Some("declaration".into()), spread: false, value: contract, expected_error: None, span });
}

fn bind(value_name: &str, value: Ex, span: Span) -> Statement {
    Statement::Binding(ast::Binding { name: value_name.into(), annotation: None, value,
        mutable: false, update: false, preserve_error: false, span })
}

fn typed_call(callee: &str, contract: ast::TypeAnnotation, values: Vec<Ex>, span: Span) -> Ex {
    let mut value = call(callee, values, span);
    if let K::Call { callee, .. } = &mut value.kind {
        *callee = Box::new(expression(K::TypeApplication { callee: callee.clone(), arguments: vec![contract] }, span));
    }
    value
}

fn receiver_check(owner: &GrammarOwner<'_>, registry_alias: &str, span: Span) -> Vec<Statement> {
    vec![
        bind("_receiver_proof", typed_call(&format!("{registry_alias}context_receiver_state"),
            annotation(owner.name, span), vec![name("context", span)], span), span),
        Statement::If {
            condition: expression(K::Binary { operator: ast::BinaryOperator::NotEqual,
                left: Box::new(name("_receiver_proof", span)),
                right: Box::new(name(&format!("{registry_alias}ProofResult.Proven"), span)) }, span),
            then_block: vec![Statement::Return { value: Some(name("_receiver_proof", span)), span }],
            else_block: Vec::new(), span,
        },
    ]
}

// Trait lexical grammars operate on atomic source tokens. Their receiver can
// be unknown during lexing, so receiver predicates are checked at selection.
// A free Lexeme capture must not greedily consume the entire remaining file.
fn lower_trait_lexical(owner: &GrammarOwner<'_>, grammar: &ast::SentenceDeclaration,
    suffix: &str, registry_alias: &str, owner_alias: &str, module: &ResolvedModule,
) -> Result<(Vec<Item>, Ex), Diagnostic> {
    let span = grammar.function.span;
    let label_name = format!("{}.{}", owner.name, grammar.function.name);
    let label = call("SyntaxDeclaration", vec![text(&module.path.to_string_lossy(), span), text(&label_name, span)], span);
    let capture_name = grammar.function.parameters.first().map_or("_spelling", |parameter| parameter.name.as_str());
    let mut prefix = String::new();
    let mut suffix_text = String::new();
    let mut captured = false;
    for field in &grammar.fields {
        match field {
            ast::SentenceElement::Literal(value) => if captured { suffix_text.push_str(value); } else { prefix.push_str(value); },
            ast::SentenceElement::Capture(_) => captured = true,
        }
    }
    let mut raw = grammar.function.clone();
    raw.name = format!("_grammar_body_{suffix}");
    raw.decorators.clear();
    raw.constraints.clear();
    raw.contracts.clear();
    raw.parameters = vec![
        ast::FunctionParameter { name: capture_name.into(), annotation: annotation(&format!("{registry_alias}Lexeme"), span),
            immutable_reference: false, variadic: false, default: None, span },
        ast::FunctionParameter { name: "context".into(), annotation: annotation(&format!("{registry_alias}GrammarContext"), span),
            immutable_reference: false, variadic: false, default: None, span },
    ];
    let receiver = bind("self", typed_call(&format!("{registry_alias}context_receiver"),
        annotation(owner.name, span), vec![name("context", span)], span), span);
    let mut body = vec![receiver.clone()];
    body.extend(grammar.function.body.clone().ok_or_else(|| Diagnostic::new("E000212", "grammar requires a construction body", Some(span)))?);
    raw.body = Some(body);
    let construct = boxed_adapter(&raw, &format!("_grammar_construct_{suffix}"), registry_alias);
    let mut accept = raw.clone();
    accept.name = format!("_grammar_accept_{suffix}");
    accept.result = annotation(&format!("{registry_alias}ProofResult"), span);
    let mut conditions = receiver_check(owner, registry_alias, span);
    conditions.push(receiver);
    for constraint in &grammar.function.constraints {
        if let ast::GenericConstraint::Predicate(predicate) = constraint {
            conditions.push(Statement::If { condition: predicate.clone(), then_block: Vec::new(),
                else_block: vec![Statement::Return { value: Some(name(&format!("{registry_alias}ProofResult.Refuted"), span)), span }], span });
        }
    }
    conditions.push(Statement::Return { value: Some(name(&format!("{registry_alias}ProofResult.Proven"), span)), span });
    accept.body = Some(conditions);
    let mut matcher = raw.clone();
    matcher.name = format!("_grammar_match_{suffix}");
    matcher.parameters[0].name = "input".into();
    matcher.parameters[0].annotation = ast::TypeAnnotation::named("list", vec![annotation(&format!("{registry_alias}TokenTerm"), span)], span);
    matcher.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![
        annotation(&format!("{registry_alias}GrammarMatch"), span), annotation("None", span), annotation(&format!("{registry_alias}Diagnostic"), span),
    ]), span };
    matcher.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}match_contextual_token"), vec![
        name("input", span), text(&label_name, span), name("context", span), name(&accept.name, span), name(&construct.name, span),
        text(&prefix, span), text(&suffix_text, span), boolean(captured, span),
    ], span)), span }]);
    let matcher_name = matcher.name.clone();
    let mut functions = vec![Item::Function(raw), Item::Function(construct), Item::Function(accept), Item::Function(matcher)];
    let mut recognizer = expression(K::Literal(Literal::None), span);
    if !captured {
        // Fixed spellings can be recognized without the receiver. Acceptance
        // still waits for the concrete trait implementation in parser context.
        let mut accepts_spelling = grammar.function.clone();
        accepts_spelling.name = format!("_grammar_spelling_{suffix}");
        accepts_spelling.decorators.clear();
        accepts_spelling.constraints.clear();
        accepts_spelling.contracts.clear();
        accepts_spelling.parameters = vec![ast::FunctionParameter { name: "spelling".into(),
            annotation: annotation(&format!("{registry_alias}Lexeme"), span), immutable_reference: false, variadic: false, default: None, span }];
        accepts_spelling.result = annotation("bool", span);
        accepts_spelling.body = Some(vec![Statement::Return { value: Some(boolean(true, span)), span }]);
        let mut recognize = accepts_spelling.clone();
        recognize.name = format!("_grammar_recognize_{suffix}");
        recognize.parameters[0].name = "input".into();
        recognize.parameters[0].annotation = annotation(&format!("{registry_alias}Window"), span);
        recognize.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![
            annotation(&format!("{registry_alias}LexicalMatch"), span), annotation("None", span), annotation(&format!("{registry_alias}Diagnostic"), span),
        ]), span };
        recognize.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}recognize_declared"), vec![
            name("input", span), call(&format!("{registry_alias}SyntaxDeclaration"), vec![text(&module.path.to_string_lossy(), span), text(&label_name, span)], span),
            text(&prefix, span), text("", span), text("", span), name(&accepts_spelling.name, span),
        ], span)), span }]);
        recognizer = name(&format!("{owner_alias}{}", recognize.name), span);
        functions.extend([Item::Function(accepts_spelling), Item::Function(recognize)]);
    }
    let result = grammar.function.result.simple_name();
    let result_value = result.map(|result| name(&format!("{}{result}", if matches!(result, "bool" | "int" | "float" | "char" | "string" | "None" | "unit" | "absent") { "" } else { owner_alias }), span)).unwrap_or_else(|| expression(K::Literal(Literal::None), span));
    let mut descriptor = call("RegisteredGrammar", vec![label.clone(), result_value, recognizer], span);
    if let K::Call { arguments, .. } = &mut descriptor.kind {
        arguments.push(ast::CallArgument { name: Some("token_form".into()), spread: false, value: boolean(true, span), expected_error: None, span });
        arguments.push(ast::CallArgument { name: Some("match_context".into()), spread: false, value: name(&format!("{owner_alias}{matcher_name}"), span), expected_error: None, span });
    }
    Ok((functions, expression(K::Tuple(vec![label, descriptor]), span)))
}

fn boxed_adapter(raw: &ast::FunctionDeclaration, adapter_name: &str, registry_alias: &str) -> ast::FunctionDeclaration {
    let span = raw.span;
    let mut adapter = raw.clone();
    adapter.name = adapter_name.into();
    adapter.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![annotation(&format!("{registry_alias}LiteralValue"), span), annotation(&format!("{registry_alias}Diagnostic"), span)]), span };
    adapter.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}literal_value"), vec![call(&raw.name, raw.parameters.iter().map(|parameter| name(&parameter.name, span)).collect(), span)], span)), span }]);
    adapter
}

// Predicate receiver reads bind to declared fields, never a fabricated `self`
// object or an invocation of the owner's construction behavior.
fn predicate_context(value: &mut Ex, owner: &GrammarOwner<'_>, registry_alias: &str) {
    if let K::Member { object, name: field_name } = &value.kind {
        if matches!(&object.kind, K::Name(value) if value == "self") {
            if let Some(field) = owner.fields.iter().find(|field| field.name == *field_name) {
                *value = typed_call(&format!("{registry_alias}context_field"), field.annotation.clone(), vec![name("context", value.span), text(field_name, value.span)], value.span);
                return;
            }
        }
    }
    match &mut value.kind {
        K::Member { object, .. } | K::Unary { operand: object, .. } | K::Await { expression: object }
        | K::Async { expression: object, .. } | K::Throw { error: object }
        | K::TypeApplication { callee: object, .. } => predicate_context(object, owner, registry_alias),
        K::Call { callee, arguments } => { predicate_context(callee, owner, registry_alias); for argument in arguments { predicate_context(&mut argument.value, owner, registry_alias); } }
        K::Binary { left, right, .. } | K::Index { object: left, index: right }
        | K::Fallback { value: left, fallback: right } => { predicate_context(left, owner, registry_alias); predicate_context(right, owner, registry_alias); }
        K::Conditional { value, condition, fallback } => { predicate_context(value, owner, registry_alias); predicate_context(condition, owner, registry_alias); predicate_context(fallback, owner, registry_alias); }
        K::List(values) | K::Set(values) | K::Tuple(values) => for value in values { predicate_context(value, owner, registry_alias); },
        K::Slice { object, start, end, step, .. } => { predicate_context(object, owner, registry_alias); for value in [start, end, step].into_iter().flatten() { predicate_context(value, owner, registry_alias); } }
        K::Map(entries) => for entry in entries { predicate_context(&mut entry.key, owner, registry_alias); predicate_context(&mut entry.value, owner, registry_alias); },
        K::Lambda { body, .. } => predicate_context(body, owner, registry_alias),
        _ => {}
    }
}

fn lower_sentence(owner: &GrammarOwner<'_>, grammar: &ast::SentenceDeclaration,
    suffix: &str, registry_alias: &str, owner_alias: &str, module: &ResolvedModule,
) -> Result<(Vec<Item>, Ex), Diagnostic> {
    let span = grammar.function.span;
    let fail = |message: &str| Diagnostic::new("E000212", message, Some(span));
    let label_name = format!("{}.{}", owner.name, grammar.function.name);
    let label = call("SyntaxDeclaration", vec![text(&module.path.to_string_lossy(), span), text(&label_name, span)], span);
    let construct_name = format!("_grammar_construct_{suffix}");
    let accept_name = format!("_grammar_accept_{suffix}");
    let match_name = format!("_grammar_match_{suffix}");
    let mut raw = grammar.function.clone();
    raw.name = format!("_grammar_body_{suffix}");
    raw.decorators.clear();
    raw.constraints.clear();
    raw.parameters = vec![
        ast::FunctionParameter { name: "captures".into(), annotation: ast::TypeAnnotation::named("list", vec![annotation(&format!("{registry_alias}CapturedGrammar"), span)], span), immutable_reference: false, variadic: false, default: None, span },
        ast::FunctionParameter { name: "context".into(), annotation: annotation(&format!("{registry_alias}GrammarContext"), span), immutable_reference: false, variadic: false, default: None, span },
    ];
    let mut bindings = Vec::new();
    let mut repetitions = Vec::new();
    for (index, parameter) in grammar.function.parameters.iter().enumerate() {
        let helper = if parameter.variadic { "capture_many" } else { "capture_one" };
        bindings.push(bind(&parameter.name, typed_call(&format!("{registry_alias}{helper}"), parameter.annotation.clone(), vec![name("captures", span), number(index, span)], span), span));
        if parameter.variadic { repetitions.push(number(index, span)); }
    }
    let mut receiver = if owner.is_trait() {
        typed_call(&format!("{registry_alias}context_receiver"), annotation(owner.name, span), vec![name("context", span)], span)
    } else { call(owner.name, Vec::new(), span) };
    if !owner.is_trait() { if let K::Call { arguments, .. } = &mut receiver.kind {
        for field in owner.fields.iter().filter(|field| field.default.is_none()) {
            let value = if let Some((index, capture)) = grammar.function.parameters.iter().enumerate().find(|(_, parameter)| parameter.name == field.name) {
                typed_call(&format!("{registry_alias}{}", if capture.variadic { "capture_many" } else { "capture_one" }), capture.annotation.clone(), vec![name("captures", span), number(index, span)], span)
            } else {
                typed_call(&format!("{registry_alias}context_field"), field.annotation.clone(), vec![name("context", span), text(&field.name, span)], span)
            };
            arguments.push(ast::CallArgument { name: Some(field.name.clone()), spread: false, value, expected_error: None, span });
        }
    } }
    let mut body = bindings.clone();
    body.push(bind("self", receiver, span));
    body.extend(grammar.function.body.clone().ok_or_else(|| fail("sentence requires a construction body"))?);
    raw.body = Some(body);
    let construct = boxed_adapter(&raw, &construct_name, registry_alias);
    let mut accepts = raw.clone();
    accepts.name = accept_name.clone();
    accepts.contracts.clear();
    accepts.hook = None;
    accepts.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![annotation(&format!("{registry_alias}ProofResult"), span), annotation(&format!("{registry_alias}Diagnostic"), span)]), span };
    let proof = |value: &str| name(&format!("{registry_alias}ProofResult.{value}"), span);
    let mut predicate = if owner.is_trait() { receiver_check(owner, registry_alias, span) } else { Vec::new() };
    predicate.push(Statement::If {
        condition: call(&format!("{registry_alias}captures_ready"), vec![name("captures", span)], span), then_block: Vec::new(),
        else_block: vec![Statement::Return { value: Some(proof("Unknown")), span }], span,
    });
    predicate.extend(bindings);
    if owner.is_trait() {
        predicate.push(bind("self", typed_call(&format!("{registry_alias}context_receiver"), annotation(owner.name, span), vec![name("context", span)], span), span));
    }
    for constraint in &grammar.function.constraints {
        let ast::GenericConstraint::Predicate(condition) = constraint else { continue; };
        let mut condition = condition.clone();
        if !owner.is_trait() { predicate_context(&mut condition, owner, registry_alias); }
        predicate.push(Statement::If { condition, then_block: Vec::new(), else_block: vec![Statement::Return { value: Some(proof("Refuted")), span }], span });
    }
    predicate.push(Statement::Return { value: Some(proof("Proven")), span });
    accepts.body = Some(predicate);
    let mut matcher = accepts.clone();
    matcher.name = match_name.clone();
    matcher.parameters.truncate(1);
    matcher.parameters[0].name = "input".into();
    matcher.parameters[0].annotation = ast::TypeAnnotation::named("list", vec![annotation(&format!("{registry_alias}TokenTerm"), span)], span);
    matcher.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![annotation(&format!("{registry_alias}GrammarMatch"), span), annotation("None", span), annotation(&format!("{registry_alias}Diagnostic"), span)]), span };
    matcher.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}match_declared_sentence"), vec![name("input", span), text(&label_name, span)], span)), span }]);
    let fields = grammar.fields.iter().map(|field| match field {
        ast::SentenceElement::Literal(value) => call("SentenceField.Literal", vec![text(value, span)], span),
        ast::SentenceElement::Capture(index) => call("SentenceField.Capture", vec![text(&grammar.function.parameters[*index].name, span), name("TermRole.Value", span), list([], span)], span),
    });
    let result = grammar.function.result.simple_name();
    let result_value = result.map(|result| name(&format!("{}{result}", if matches!(result, "bool" | "int" | "float" | "char" | "string" | "None") { "" } else { owner_alias }), span)).unwrap_or_else(|| expression(K::Literal(Literal::None), span));
    let capture_types = grammar.function.parameters.iter().map(|parameter| {
        let contract = &parameter.annotation;
        let Some(value) = contract.simple_name() else {
            return Err(Diagnostic::new("E000212", "sentence capture requires specialization before an executable adapter can be generated", Some(contract.span)));
        };
        Ok(name(&format!("{}{value}", if matches!(value, "bool" | "int" | "float" | "char" | "string" | "None" | "unit" | "absent") { "" } else { owner_alias }), contract.span))
    }).collect::<Result<Vec<_>, Diagnostic>>()?;
    let descriptor = call("RegisteredGrammar", vec![label.clone(), result_value,
        expression(K::Literal(Literal::None), span), list(fields, span), name(&format!("{owner_alias}{match_name}"), span),
        boolean(false, span), list(capture_types, span), list(repetitions, span), name(&format!("{owner_alias}{accept_name}"), span), name(&format!("{owner_alias}{construct_name}"), span)], span);
    Ok((vec![Item::Function(raw), Item::Function(construct), Item::Function(accepts), Item::Function(matcher)], expression(K::Tuple(vec![label, descriptor]), span)))
}

/// Materialize the source registry's initializer and recognition functions in
/// the owners' lexical scopes. No grammar construction body is executed here.
pub(super) fn lower_registrations(graph: &ModuleGraph) -> Result<ModuleGraph, Diagnostic> {
    let registries: Vec<_> = graph.modules.iter().enumerate().filter(|(_, module)| {
        module.ast.items.iter().any(|item| matches!(item, Item::Function(function)
            if function.decorators.iter().any(|decorator| decorator.name == "compiler_grammar_registry")))
    }).collect();
    let Some(&(registry_index, registry_module)) = registries.first() else {
        return Ok(graph.clone());
    };
    if registries.len() != 1 {
        return Err(Diagnostic::new("E000203", "multiple compiler grammar registries", None));
    }
    let mut output = graph.clone();
    let mut entries = Vec::new();
    for (module_index, module) in graph.modules.iter().enumerate() {
        let registry_alias = if module_index == registry_index { String::new() } else { "_grammar_registry.".into() };
        let owner_alias = if module_index == registry_index { String::new() } else { format!("_grammar_owner_{:032x}.", module.id.0) };
        let mut generated = Vec::new();
        let mut first_span = None;
        for item in &module.ast.items {
            let Some(owner) = GrammarOwner::from_item(item) else { continue; };
            for (ordinal, grammar) in owner.sentences.iter().enumerate() {
                let span = grammar.function.span;
                first_span.get_or_insert(span);
                let fail = |message: &str| Diagnostic::new("E000212", message, Some(span));
                let contract = grammar_contract(module, &owner, grammar, ordinal)?;
                if !supports_legacy_adapter(&owner, grammar) {
                    let label = call("SyntaxDeclaration", vec![text(&module.path.to_string_lossy(), span), text(&format!("{}.{}", owner.name, grammar.function.name), span)], span);
                    let mut descriptor = call("RegisteredGrammar", vec![label.clone(), expression(K::Literal(Literal::None), span), expression(K::Literal(Literal::None), span)], span);
                    if let K::Call { arguments, .. } = &mut descriptor.kind {
                        arguments.push(ast::CallArgument { name: Some("token_form".into()), spread: false, value: boolean(grammar.lexical, span), expected_error: None, span });
                    }
                    let mut entry = expression(K::Tuple(vec![label, descriptor]), span);
                    attach_contract(&mut entry, contract);
                    entries.push(entry);
                    continue;
                }
                if !grammar.lexical {
                    let suffix = format!("{:032x}_{:032x}_{ordinal}", module.id.0, super::stable_hash(&owner.name));
                    let (functions, mut entry) = lower_sentence(&owner, grammar, &suffix, &registry_alias, &owner_alias, module)?;
                    attach_contract(&mut entry, contract);
                    generated.extend(functions);
                    entries.push(entry);
                    continue;
                }
                if owner.is_trait() {
                    let suffix = format!("{:032x}_{:032x}_{ordinal}", module.id.0, super::stable_hash(owner.name));
                    let (functions, mut entry) = lower_trait_lexical(&owner, grammar, &suffix, &registry_alias, &owner_alias, module)?;
                    attach_contract(&mut entry, contract);
                    generated.extend(functions);
                    entries.push(entry);
                    continue;
                }
                let captures: Vec<_> = grammar.fields.iter().filter_map(|field| match field {
                    ast::SentenceElement::Capture(index) => Some(*index),
                    _ => None,
                }).collect();
                if captures.len() > 1 || grammar.function.parameters.len() != captures.len()
                    || grammar.function.parameters.iter().any(|parameter| parameter.variadic || parameter.annotation.simple_name() != Some("Lexeme")) {
                    return Err(fail("automatic lexical registration currently requires fixed spellings or one non-repeated Lexeme capture"));
                }
                if !owner.type_parameters.is_empty() && grammar.function.constraints.iter().any(|constraint| matches!(constraint, ast::GenericConstraint::Parameter { .. })) {
                    return Err(fail("grammar type constraints require specialization before registration"));
                }
                let result = grammar.function.result.simple_name().ok_or_else(|| fail("grammar result must be a named contract for automatic registration"))?;
                let suffix = format!("{:032x}_{:032x}_{ordinal}", module.id.0, super::stable_hash(&owner.name));
                let accept_name = format!("_grammar_accept_{suffix}");
                let recognize_name = format!("_grammar_recognize_{suffix}");
                let construct_name = format!("_grammar_construct_{suffix}");
                let match_name = format!("_grammar_match_{suffix}");
                let capture_name = grammar.function.parameters.first().map(|parameter| parameter.name.as_str()).unwrap_or("_spelling");
                let mut accept = grammar.function.clone();
                accept.name = accept_name.clone();
                accept.decorators.clear();
                accept.type_parameters.clear();
                accept.constraints.clear();
                accept.contracts.clear();
                accept.hook = None;
                accept.parameters = vec![ast::FunctionParameter {
                    name: capture_name.into(), annotation: annotation(&format!("{registry_alias}Lexeme"), span),
                    immutable_reference: false, variadic: false, default: None, span,
                }];
                accept.result = annotation("bool", span);
                let mut body = Vec::new();
                for constraint in &grammar.function.constraints {
                    let ast::GenericConstraint::Predicate(predicate) = constraint else {
                        return Err(fail("grammar constraint cannot be evaluated from its lexical capture"));
                    };
                    body.push(Statement::If {
                        condition: predicate.clone(), then_block: Vec::new(),
                        else_block: vec![Statement::Return { value: Some(expression(K::Literal(Literal::Boolean(false)), span)), span }], span,
                    });
                }
                body.push(Statement::Return { value: Some(expression(K::Literal(Literal::Boolean(true)), span)), span });
                accept.body = Some(body);
                let mut prefix = String::new();
                let mut suffix_text = String::new();
                let mut after_capture = false;
                let mut fields = Vec::new();
                for field in &grammar.fields {
                    match field {
                        ast::SentenceElement::Literal(spelling) => {
                            if after_capture { suffix_text.push_str(spelling); } else { prefix.push_str(spelling); }
                            fields.push(call("SentenceField.Literal", vec![text(spelling, span)], span));
                        }
                        ast::SentenceElement::Capture(index) => {
                            if *index >= grammar.function.parameters.len() { return Err(fail("grammar capture index is invalid")); }
                            after_capture = true;
                            fields.push(call("SentenceField.Capture", vec![text(capture_name, span), name("TermRole.Symbol", span), expression(K::List(Vec::new()), span)], span));
                        }
                    }
                }
                let label_name = format!("{}.{}", owner.name, grammar.function.name);
                let label = |prefix: &str| call(&format!("{prefix}SyntaxDeclaration"), vec![text(&module.path.to_string_lossy(), span), text(&label_name, span)], span);
                let mut recognize = accept.clone();
                recognize.name = recognize_name.clone();
                recognize.parameters[0].name = "input".into();
                recognize.parameters[0].annotation = annotation(&format!("{registry_alias}Window"), span);
                recognize.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![
                    annotation(&format!("{registry_alias}LexicalMatch"), span), annotation("None", span), annotation(&format!("{registry_alias}Diagnostic"), span),
                ]), span };
                recognize.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}recognize_declared"), vec![
                    name("input", span), label(&registry_alias), text(&prefix, span), text(&suffix_text, span),
                    text(if after_capture { capture_name } else { "" }, span), name(&accept_name, span),
                ], span)), span }]);
                // Construction is a distinct callable: probing a token never runs it.
                let mut construct = accept.clone();
                construct.name = format!("_grammar_body_{suffix}");
                construct.result = grammar.function.result.clone();
                construct.body = grammar.function.body.clone();
                let mut matcher = recognize.clone();
                matcher.name = match_name.clone();
                matcher.parameters[0].annotation = ast::TypeAnnotation::named("list", vec![annotation(&format!("{registry_alias}TokenTerm"), span)], span);
                matcher.result = ast::TypeAnnotation { kind: ast::TypeAnnotationKind::Union(vec![
                    annotation(&format!("{registry_alias}GrammarMatch"), span), annotation("None", span), annotation(&format!("{registry_alias}Diagnostic"), span),
                ]), span };
                matcher.body = Some(vec![Statement::Return { value: Some(call(&format!("{registry_alias}match_declared_token"), vec![
                    name("input", span), text(&label_name, span), name(&construct_name, span),
                    text(&prefix, span), text(&suffix_text, span), expression(K::Literal(Literal::Boolean(after_capture)), span),
                ], span)), span }]);
                let adapter = boxed_adapter(&construct, &construct_name, &registry_alias);
                generated.extend([Item::Function(accept), Item::Function(recognize), Item::Function(construct), Item::Function(adapter), Item::Function(matcher)]);
                let result_name = match result {
                    "bool" | "int" | "float" | "char" | "string" | "None" | "absent" | "unit" => result.to_owned(),
                    _ => format!("{owner_alias}{result}"),
                };
                let descriptor = call("RegisteredGrammar", vec![label(""), name(&result_name, span), name(&format!("{owner_alias}{recognize_name}"), span), expression(K::List(fields), span), name(&format!("{owner_alias}{match_name}"), span), expression(K::Literal(Literal::Boolean(true)), span)], span);
                let mut entry = expression(K::Tuple(vec![label(""), descriptor]), span);
                attach_contract(&mut entry, contract);
                entries.push(entry);
            }
        }
        if let Some(span) = first_span {
            if module_index != registry_index {
                import(&mut output.modules[module_index], registry_module, "_grammar_registry", span);
                import(&mut output.modules[registry_index], module, owner_alias.trim_end_matches('.'), span);
            }
            output.modules[module_index].ast.items.extend(generated);
        }
    }
    let binding = output.modules[registry_index].ast.items.iter_mut().find_map(|item| match item {
        Item::Binding(binding) if binding.name == "registered_grammars" => Some(binding),
        _ => None,
    }).ok_or_else(|| Diagnostic::new("E000212", "grammar registry has no declaration table initializer", None))?;
    let K::List(existing) = &mut binding.value.kind else {
        return Err(Diagnostic::new("E000212", "grammar registry initializer must be a list", Some(binding.span)));
    };
    existing.extend(entries);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_modules::{ModuleId, PackageId};
    use severian_source::SourceFile;

    fn module(id: u128, source: &str) -> ResolvedModule {
        let source = SourceFile::virtual_source(format!("module-{id}.sev"), source);
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        ResolvedModule { id: ModuleId(id), package: PackageId(0), path: format!("module-{id}.sev").into(), source, ast, imports: Vec::new() }
    }

    fn graph(owner: &str) -> ModuleGraph {
        ModuleGraph { policies: Default::default(), modules: vec![
            module(1, "registered_grammars = []\n@compiler_grammar_registry\ndef registry():\n    return registered_grammars\n"),
            module(2, owner),
        ] }
    }

    fn descriptors(graph: &ModuleGraph) -> Vec<&[ast::CallArgument]> {
        let table = graph.modules[0].ast.items.iter().find_map(|item| match item {
            Item::Binding(binding) if binding.name == "registered_grammars" => Some(&binding.value), _ => None,
        }).unwrap();
        let K::List(entries) = &table.kind else { panic!("registry") };
        entries.iter().map(|entry| {
            let K::Tuple(pair) = &entry.kind else { panic!("entry") };
            let K::Call { arguments, .. } = &pair[1].kind else { panic!("descriptor") };
            arguments.as_slice()
        }).collect()
    }

    #[test]
    fn trait_grammar_keeps_identity_and_defers_receiver_dependent_recognition() {
        let original = graph("trait B:\n    indentation_unit: string = \"    \"\n    grammar indentation[spelling: Lexeme]() -> string with spelling.text == self.indentation_unit:\n        return construction_must_not_run(spelling.text)\ntrait Derived: B\nclass Block: Derived\n    indentation_unit: string = \"\\t\"\n");
        let lowered = lower_registrations(&original).unwrap();
        let descriptors = descriptors(&lowered);
        // Inheriting a grammar doesn't duplicate its declaration.
        assert_eq!(descriptors.len(), 1);
        let descriptor = descriptors[0];
        assert!(matches!(descriptor[2].value.kind, K::Literal(Literal::None)), "an unbounded capture cannot become a greedy lexer recognizer");
        assert!(descriptor.iter().any(|argument| argument.name.as_deref() == Some("match_context")));
        let contract = &descriptor.iter().find(|argument| argument.name.as_deref() == Some("declaration")).unwrap().value;
        let K::Call { arguments, .. } = &contract.kind else { panic!("contract") };
        let K::Call { arguments: callable, .. } = &arguments[0].value.kind else { panic!("callable") };
        let K::Call { arguments: owner, .. } = &callable[1].value.kind else { panic!("owner") };
        let K::Call { arguments: identity, .. } = &owner[2].value.kind else { panic!("identity") };
        assert!(matches!(&identity[0].value.kind, K::Literal(Literal::Integer(value)) if value == &super::super::stable_hash("trait:B").to_string()));
        let functions: Vec<_> = lowered.modules[1].ast.items.iter().filter_map(|item| match item { Item::Function(f) => Some(f), _ => None }).collect();
        let accept = functions.iter().find(|function| function.name.starts_with("_grammar_accept_")).unwrap();
        let body = accept.body.as_ref().unwrap();
        assert!(matches!(&body[0], Statement::Binding(binding) if binding.name == "_receiver_proof"));
        assert!(matches!(&body[1], Statement::If { then_block, .. } if matches!(&then_block[0], Statement::Return { value: Some(Ex { kind: K::Name(value), .. }), .. } if value == "_receiver_proof")));
        assert!(matches!(&body[2], Statement::Binding(binding) if binding.name == "self"));
        assert!(functions.iter().filter(|function| !function.name.starts_with("_grammar_body_")).all(|function| !format!("{:?}", function.body).contains("construction_must_not_run")));
        let indexed = super::super::collect_declarations(&original).unwrap();
        assert!(matches!(&indexed.grammars[0].owner_declaration, super::super::GrammarOwnerDeclaration::Trait(owner) if owner.name == "B"));
    }

    #[test]
    fn trait_grammar_fixed_spelling_has_a_pure_recognizer_and_contextual_matcher() {
        let original = graph("trait Mark:\n    enabled: bool\n    grammar marker[\"@@\"]() -> int with self.enabled:\n        return construct_marker()\n");
        let lowered = lower_registrations(&original).unwrap();
        let descriptor = descriptors(&lowered)[0];
        assert!(matches!(descriptor[2].value.kind, K::Name(_)));
        assert!(descriptor.iter().any(|argument| argument.name.as_deref() == Some("match_context")));
        let functions: Vec<_> = lowered.modules[1].ast.items.iter().filter_map(|item| match item { Item::Function(f) => Some(f), _ => None }).collect();
        let spelling = functions.iter().find(|function| function.name.starts_with("_grammar_spelling_")).unwrap();
        assert!(matches!(spelling.body.as_deref().unwrap(), [Statement::Return { value: Some(Ex { kind: K::Literal(Literal::Boolean(true)), .. }), .. }]));
        let accept = functions.iter().find(|function| function.name.starts_with("_grammar_accept_")).unwrap();
        assert!(format!("{:?}", accept.body).contains("enabled"));
        assert!(!format!("{:?}", accept.body).contains("construct_marker"));
    }

    #[test]
    fn trait_sentence_uses_context_receiver_instead_of_constructing_the_trait() {
        let original = graph("trait Branch:\n    enabled: bool\n    sentence branch[\"branch\", value: int]() -> Branch with self.enabled:\n        return self\n");
        let lowered = lower_registrations(&original).unwrap();
        let raw = lowered.modules[1].ast.items.iter().find_map(|item| match item {
            Item::Function(f) if f.name.starts_with("_grammar_body_") => Some(f), _ => None,
        }).unwrap();
        let receiver = raw.body.as_ref().unwrap().iter().find_map(|statement| match statement {
            Statement::Binding(binding) if binding.name == "self" => Some(&binding.value), _ => None,
        }).unwrap();
        let K::Call { callee, .. } = &receiver.kind else { panic!("receiver access") };
        let K::TypeApplication { callee, arguments } = &callee.kind else { panic!("typed receiver") };
        assert!(matches!(&callee.kind, K::Name(value) if value.ends_with("context_receiver")));
        assert_eq!(arguments[0].simple_name(), Some("Branch"));
    }

    #[test]
    fn generic_trait_grammar_remains_discoverable_without_an_unsafe_adapter() {
        let original = graph("trait Provider[T = int]:\n    grammar value[spelling: Lexeme]() -> T:\n        return self.convert(spelling)\n");
        let lowered = lower_registrations(&original).unwrap();
        let descriptor = descriptors(&lowered)[0];
        assert!(matches!(descriptor[2].value.kind, K::Literal(Literal::None)));
        let contract = descriptor.iter().find(|argument| argument.name.as_deref() == Some("declaration")).unwrap();
        let K::Call { arguments, .. } = &contract.value.kind else { panic!("contract") };
        let K::List(defaults) = &arguments[8].value.kind else { panic!("owner defaults") };
        assert_eq!(defaults.len(), 1);
        assert!(!matches!(defaults[0].kind, K::Literal(Literal::None)));
        assert!(lowered.modules[1].ast.items.iter().all(|item| !matches!(item, Item::Function(_))));
    }

    #[test]
    fn compilation_populates_registry_and_generates_recognition_without_construction() {
        let original = graph("class Example:\n    grammar fixed[\"example\"]() -> bool:\n        return construction_must_not_run()\n    grammar captured[value: Lexeme]() -> int with valid(value.text):\n        return construction_must_not_run(value)\n    sentence semantic[\"later\"]() -> B:\n        return self\n");
        let lowered = lower_registrations(&original).unwrap();
        let table = lowered.modules[0].ast.items.iter().find_map(|item| match item {
            Item::Binding(binding) if binding.name == "registered_grammars" => Some(&binding.value),
            _ => None,
        }).unwrap();
        let K::List(entries) = &table.kind else { panic!("expected registry entries") };
        assert_eq!(entries.len(), 3);
        let functions: Vec<_> = lowered.modules[1].ast.items.iter().filter_map(|item| match item {
            Item::Function(function) => Some(function), _ => None,
        }).collect();
        assert!(functions.iter().filter(|function| !function.name.starts_with("_grammar_body_")).all(|function| !format!("{:?}", function.body).contains("construction_must_not_run")));
        let predicate = functions.iter().find(|function| function.name.starts_with("_grammar_accept_") && function.parameters[0].name == "value").unwrap();
        assert!(matches!(predicate.body.as_ref().unwrap()[0], Statement::If { .. }));
        let constructors: Vec<_> = functions.iter().filter(|function| function.name.starts_with("_grammar_construct_")).collect();
        assert_eq!(constructors.len(), 3);
        assert_eq!(functions.iter().filter(|function| format!("{:?}", function.body).contains("construction_must_not_run")).count(), 2);
        assert!(constructors.iter().all(|function| format!("{:?}", function.body).contains("_grammar_body_")));
        assert_eq!(lowered.modules[0].imports.len(), 1);
        assert_eq!(lowered.modules[1].imports.len(), 1);
        for entry in entries {
            let K::Tuple(pair) = &entry.kind else { panic!("expected label and descriptor") };
            let K::Call { arguments, .. } = &pair[1].kind else { panic!("expected descriptor") };

            let K::Name(parser_callable) = &arguments[4].value.kind else { panic!("expected parser callable") };
            assert!(functions.iter().any(|function| parser_callable.ends_with(&format!(".{}", function.name))));
            if let K::Name(callable) = &arguments[2].value.kind {
                assert!(functions.iter().any(|function| callable.ends_with(&format!(".{}", function.name))));
            } else {
                assert!(matches!(arguments[5].value.kind, K::Literal(Literal::Boolean(false))));
            }
        }
    }

    #[test]
    fn sentence_registration_retains_multiple_typed_and_repeated_captures() {
        let original = graph("class Branch:\n    parent: B\n    body: B\n    sentence choose[\"choose\", first: bool, rest...: int, \":\"]() -> B with len(rest) > 0:\n        return self\n");
        let lowered = lower_registrations(&original).unwrap();
        let table = lowered.modules[0].ast.items.iter().find_map(|item| match item { Item::Binding(binding) if binding.name == "registered_grammars" => Some(&binding.value), _ => None }).unwrap();
        let K::List(entries) = &table.kind else { panic!("registry") };
        let K::Tuple(pair) = &entries[0].kind else { panic!("entry") };
        let K::Call { arguments, .. } = &pair[1].kind else { panic!("descriptor") };
        let K::List(fields) = &arguments[3].value.kind else { panic!("sequence") };
        assert_eq!(fields.len(), 4);
        let K::List(types) = &arguments[6].value.kind else { panic!("types") };
        assert_eq!(types, &vec![name("bool", types[0].span), name("int", types[1].span)]);
        let K::List(repeated) = &arguments[7].value.kind else { panic!("repetition") };
        assert!(matches!(&repeated[0].kind, K::Literal(Literal::Integer(index)) if index == "1"));
        let functions: Vec<_> = lowered.modules[1].ast.items.iter().filter_map(|item| match item { Item::Function(function) => Some(function), _ => None }).collect();
        let accept = functions.iter().find(|function| function.name.starts_with("_grammar_accept_")).unwrap();
        assert!(accept.body.as_ref().unwrap().iter().any(|statement| matches!(statement, Statement::If { .. })));
        let construct = functions.iter().find(|function| function.name.starts_with("_grammar_body_")).unwrap();
        assert_eq!(construct.result.simple_name(), Some("B"));
        assert!(format!("{:?}", construct.body).contains("capture_many"));
    }

    #[test]
    fn generated_token_matcher_source_parses() {
        module(3, include_str!("../../../../../sev_compiler/frontend/parser/match.sev"));
        module(7, include_str!("../../../../../sev_compiler/frontend/parser/parser.sev"));
        module(5, include_str!("../../../../../sev_compiler/frontend/parser/sequence.sev"));
        module(6, include_str!("../../../../../sev_compiler/syntax/grammar/resolution.sev"));
        module(4, include_str!("../../../../../sev_compiler/frontend/parser/contract.sev"));
    }

    #[test]
    fn capture_without_legacy_adapter_retains_its_declaration_contract() {
        let source = graph("class Example:\n    grammar typed[value: int]() -> int:\n        return value\n");
        let lowered = lower_registrations(&source).unwrap();
        let descriptors = descriptors(&lowered);
        assert_eq!(descriptors.len(), 1);
        assert!(matches!(descriptors[0][2].value.kind, K::Literal(Literal::None)));
        assert!(descriptors[0].iter().any(|argument| argument.name.as_deref() == Some("declaration")));
        assert!(lowered.modules[1].ast.items.iter().all(|item| !matches!(item, Item::Function(_))));
    }
}
