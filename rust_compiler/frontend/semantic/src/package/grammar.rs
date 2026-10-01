use severian_ast::{self as ast, Expression as Ex, ExpressionKind as K, Item, Literal, Statement};
use severian_diagnostics::Diagnostic;
use severian_modules::{ModuleGraph, ResolvedImport, ResolvedModule};
use severian_source::Span;

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
            let Item::Class(owner) = item else { continue; };
            for (ordinal, grammar) in owner.sentences.iter().enumerate().filter(|(_, declaration)| declaration.lexical) {
                let span = grammar.function.span;
                first_span.get_or_insert(span);
                let fail = |message: &str| Diagnostic::new("E000212", message, Some(span));
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
                generated.extend([Item::Function(accept), Item::Function(recognize)]);
                let result_name = match result {
                    "bool" | "int" | "float" | "char" | "string" | "None" | "absent" | "unit" => result.to_owned(),
                    _ => format!("{owner_alias}{result}"),
                };
                let descriptor = call("RegisteredGrammar", vec![label(""), name(&result_name, span), name(&format!("{owner_alias}{recognize_name}"), span), expression(K::List(fields), span), expression(K::Literal(Literal::None), span), expression(K::Literal(Literal::Boolean(true)), span)], span);
                entries.push(expression(K::Tuple(vec![label(""), descriptor]), span));
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

    #[test]
    fn compilation_populates_registry_and_generates_recognition_without_construction() {
        let original = graph("class Example:\n    grammar fixed[\"example\"]() -> bool:\n        return construction_must_not_run()\n    grammar captured[value: Lexeme]() -> int with valid(value.text):\n        return construction_must_not_run(value)\n    sentence semantic[\"later\"]() -> B:\n        return self\n");
        let lowered = lower_registrations(&original).unwrap();
        let table = lowered.modules[0].ast.items.iter().find_map(|item| match item {
            Item::Binding(binding) if binding.name == "registered_grammars" => Some(&binding.value),
            _ => None,
        }).unwrap();
        let K::List(entries) = &table.kind else { panic!("expected registry entries") };
        assert_eq!(entries.len(), 2);
        let functions: Vec<_> = lowered.modules[1].ast.items.iter().filter_map(|item| match item {
            Item::Function(function) => Some(function), _ => None,
        }).collect();
        assert_eq!(functions.len(), 4);
        assert!(functions.iter().all(|function| !format!("{:?}", function.body).contains("construction_must_not_run")));
        let predicate = functions.iter().find(|function| function.name.starts_with("_grammar_accept_") && function.parameters[0].name == "value").unwrap();
        assert!(matches!(predicate.body.as_ref().unwrap()[0], Statement::If { .. }));
        assert_eq!(lowered.modules[0].imports.len(), 1);
        assert_eq!(lowered.modules[1].imports.len(), 1);
        for entry in entries {
            let K::Tuple(pair) = &entry.kind else { panic!("expected label and descriptor") };
            let K::Call { arguments, .. } = &pair[1].kind else { panic!("expected descriptor") };
            assert!(matches!(arguments[5].value.kind, K::Literal(Literal::Boolean(true))));
            let K::Name(callable) = &arguments[2].value.kind else { panic!("expected recognition callable") };
            assert!(functions.iter().any(|function| callable.ends_with(&format!(".{}", function.name))));
        }
    }

    #[test]
    fn unsupported_capture_is_diagnosed_instead_of_registered_without_its_contract() {
        let source = graph("class Example:\n    grammar typed[value: int]() -> int:\n        return value\n");
        let error = lower_registrations(&source).unwrap_err();
        assert!(error.message.contains("Lexeme capture"));
        assert!(error.span.is_some());
    }
}
