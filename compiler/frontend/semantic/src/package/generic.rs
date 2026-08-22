use super::*;

pub(super) type Substitution = BTreeMap<String, String>;

pub(super) fn collect_generic_specializations(
    module_graph: &ModuleGraph,
    index: &ProgramIndex,
) -> Result<BTreeMap<DefId, Substitution>, Diagnostic> {
    let mut specializations = BTreeMap::new();
    // A specialization can make calls inside another generic body concrete,
    // so walk to a fixed point. This is declaration-driven and independent of
    // module initialization order.
    loop {
        let before = specializations.len();
        for module in &module_graph.modules {
            let mut globals = BTreeMap::new();
            for item in &module.ast.items {
                match item {
                    Item::Binding(binding) => {
                        let expected = binding.annotation.as_ref().and_then(simple_type_name);
                        visit_expression_for_specializations(
                            module.id,
                            &binding.value,
                            expected.as_deref(),
                            &globals,
                            index,
                            &mut specializations,
                        )?;
                        if let Some(ty) =
                            expected.or_else(|| expression_type_name(&binding.value, &globals))
                        {
                            globals.insert(binding.name.clone(), ty);
                        }
                    }
                    Item::Expression(expression) => visit_expression_for_specializations(
                        module.id,
                        expression,
                        None,
                        &globals,
                        index,
                        &mut specializations,
                    )?,
                    Item::Function(function) => {
                        let id = function_def_id(module.package, module.id, function);
                        let substitution = if function.type_parameters.is_empty() {
                            Some(Substitution::new())
                        } else {
                            specializations.get(&id).cloned()
                        };
                        let Some(substitution) = substitution else {
                            continue;
                        };
                        let mut names = globals.clone();
                        for parameter in &function.parameters {
                            if let Some(name) =
                                specialized_type_name(&parameter.annotation, &substitution)
                            {
                                names.insert(parameter.name.clone(), name);
                            }
                        }
                        let result = specialized_type_name(&function.result, &substitution);
                        if let Some(body) = &function.body {
                            visit_statements_for_specializations(
                                module.id,
                                body,
                                result.as_deref(),
                                &mut names,
                                index,
                                &mut specializations,
                            )?;
                        }
                    }
                    _ => {}
                }
            }
        }
        if specializations.len() == before {
            break;
        }
    }
    Ok(specializations)
}

fn visit_statements_for_specializations(
    module: ModuleId,
    statements: &[severian_ast::Statement],
    result: Option<&str>,
    names: &mut BTreeMap<String, String>,
    index: &ProgramIndex,
    specializations: &mut BTreeMap<DefId, Substitution>,
) -> Result<(), Diagnostic> {
    for statement in statements {
        match statement {
            severian_ast::Statement::Binding(binding) => {
                let expected = binding.annotation.as_ref().and_then(simple_type_name);
                visit_expression_for_specializations(
                    module,
                    &binding.value,
                    expected.as_deref(),
                    names,
                    index,
                    specializations,
                )?;
                if let Some(ty) = expected.or_else(|| expression_type_name(&binding.value, names)) {
                    names.insert(binding.name.clone(), ty);
                }
            }
            severian_ast::Statement::Expression(expression) => {
                visit_expression_for_specializations(
                    module,
                    expression,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Return {
                value: Some(value), ..
            } => {
                visit_expression_for_specializations(
                    module,
                    value,
                    result,
                    names,
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Return { value: None, .. } => {}
            severian_ast::Statement::Assert {
                condition, message, ..
            } => {
                visit_expression_for_specializations(
                    module,
                    condition,
                    Some("bool"),
                    names,
                    index,
                    specializations,
                )?;
                if let Some(message) = message {
                    visit_expression_for_specializations(
                        module,
                        message,
                        Some("string"),
                        names,
                        index,
                        specializations,
                    )?;
                }
            }
            severian_ast::Statement::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                visit_expression_for_specializations(
                    module,
                    condition,
                    Some("bool"),
                    names,
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    then_block,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
                visit_statements_for_specializations(
                    module,
                    else_block,
                    result,
                    &mut names.clone(),
                    index,
                    specializations,
                )?;
            }
            severian_ast::Statement::Match { subject, cases, .. } => {
                visit_expression_for_specializations(
                    module,
                    subject,
                    None,
                    names,
                    index,
                    specializations,
                )?;
                for case in cases {
                    let mut case_names = names.clone();
                    if let (Some(binding), Some(annotation)) = (&case.binding, &case.annotation) {
                        if let Some(ty) = simple_type_name(annotation) {
                            case_names.insert(binding.clone(), ty);
                        }
                    }
                    visit_statements_for_specializations(
                        module,
                        &case.body,
                        result,
                        &mut case_names,
                        index,
                        specializations,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn visit_expression_for_specializations(
    module: ModuleId,
    expression: &severian_ast::Expression,
    expected: Option<&str>,
    names: &BTreeMap<String, String>,
    index: &ProgramIndex,
    specializations: &mut BTreeMap<DefId, Substitution>,
) -> Result<(), Diagnostic> {
    match &expression.kind {
        severian_ast::ExpressionKind::Call { callee, arguments } => {
            if let Some(path) = ast_callable_path(callee) {
                for definition in resolve_path(module, &path, index) {
                    let DefKind::Function(signature) = &index.definitions[&definition].kind else {
                        continue;
                    };
                    if signature.type_parameters.is_empty()
                        || signature.parameters.len() != arguments.len()
                    {
                        continue;
                    }
                    let mut substitution = Substitution::new();
                    if let Some(expected) = expected {
                        infer_substitution(
                            &signature.result,
                            expected,
                            &signature.type_parameters,
                            &mut substitution,
                        );
                    }
                    for (parameter, argument) in signature.parameters.iter().zip(arguments) {
                        if let Some(actual) = expression_type_name(argument, names) {
                            infer_substitution(
                                parameter,
                                &actual,
                                &signature.type_parameters,
                                &mut substitution,
                            );
                        }
                    }
                    if signature
                        .type_parameters
                        .iter()
                        .all(|parameter| substitution.contains_key(parameter))
                    {
                        if let Some(existing) = specializations.get(&definition) {
                            if existing != &substitution {
                                return Err(Diagnostic::new(
                                    "E000204",
                                    format!(
                                        "generic function `{}` requires more than one concrete instance; multi-instance monomorphization is not implemented yet",
                                        index.definitions[&definition].name
                                    ),
                                    Some(expression.span),
                                ));
                            }
                        } else {
                            specializations.insert(definition, substitution);
                        }
                    }
                }
            }
            for argument in arguments {
                visit_expression_for_specializations(
                    module,
                    argument,
                    None,
                    names,
                    index,
                    specializations,
                )?;
            }
        }
        severian_ast::ExpressionKind::Member { object, .. } => {
            visit_expression_for_specializations(
                module,
                object,
                None,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Unary { operand, .. } => {
            visit_expression_for_specializations(
                module,
                operand,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Binary { left, right, .. } => {
            visit_expression_for_specializations(
                module,
                left,
                expected,
                names,
                index,
                specializations,
            )?;
            visit_expression_for_specializations(
                module,
                right,
                expected,
                names,
                index,
                specializations,
            )?;
        }
        severian_ast::ExpressionKind::Literal(_) | severian_ast::ExpressionKind::Name(_) => {}
    }
    Ok(())
}

fn resolve_path(module: ModuleId, path: &str, index: &ProgramIndex) -> Vec<DefId> {
    let mut parts = path.split('.');
    let Some(first) = parts.next() else {
        return Vec::new();
    };
    let Some(mut resolution) = index.modules[&module].scope.bindings.get(first) else {
        return Vec::new();
    };
    for part in parts {
        let Resolution::Module(target) = resolution else {
            return Vec::new();
        };
        let Some(next) = index
            .exports
            .get(target)
            .and_then(|exports| exports.get(part))
        else {
            return Vec::new();
        };
        resolution = next;
    }
    resolution_definitions(resolution)
}

fn ast_callable_path(expression: &severian_ast::Expression) -> Option<String> {
    match &expression.kind {
        severian_ast::ExpressionKind::Name(name) => Some(name.clone()),
        severian_ast::ExpressionKind::Member { object, name } => {
            Some(format!("{}.{}", ast_callable_path(object)?, name))
        }
        _ => None,
    }
}

fn infer_substitution(
    pattern: &TypeAnnotation,
    actual: &str,
    parameters: &[String],
    substitution: &mut Substitution,
) {
    if let Some(name) = pattern.simple_name() {
        if parameters.iter().any(|parameter| parameter == name) {
            substitution
                .entry(name.to_owned())
                .or_insert_with(|| actual.to_owned());
        }
    }
}

fn expression_type_name(
    expression: &severian_ast::Expression,
    names: &BTreeMap<String, String>,
) -> Option<String> {
    match &expression.kind {
        severian_ast::ExpressionKind::Name(name) => names.get(name).cloned(),
        severian_ast::ExpressionKind::Literal(literal) => Some(
            match literal {
                severian_ast::Literal::Integer(_) => "int",
                severian_ast::Literal::Float(_) => "float",
                severian_ast::Literal::Boolean(_) => "bool",
                severian_ast::Literal::Character(_) => "char",
                severian_ast::Literal::String(_) => "string",
                severian_ast::Literal::Bytes(_) => "bytes",
                severian_ast::Literal::None => "none",
                severian_ast::Literal::Unit => "unit",
            }
            .to_owned(),
        ),
        _ => None,
    }
}

fn simple_type_name(annotation: &TypeAnnotation) -> Option<String> {
    annotation.simple_name().map(str::to_owned)
}

fn specialized_type_name(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> Option<String> {
    let name = annotation.simple_name()?;
    Some(
        substitution
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned()),
    )
}

pub(super) fn specialize_function(
    function: &severian_ast::FunctionDeclaration,
    substitution: &Substitution,
) -> severian_ast::FunctionDeclaration {
    let mut function = function.clone();
    function.type_parameters.clear();
    for parameter in &mut function.parameters {
        parameter.annotation = specialize_annotation(&parameter.annotation, substitution);
    }
    function.result = specialize_annotation(&function.result, substitution);
    function
}

pub(super) fn specialize_signature(
    function: &FunctionDecl,
    substitution: &Substitution,
) -> FunctionDecl {
    FunctionDecl {
        type_parameters: Vec::new(),
        parameters: function
            .parameters
            .iter()
            .map(|annotation| specialize_annotation(annotation, substitution))
            .collect(),
        result: specialize_annotation(&function.result, substitution),
    }
}

fn specialize_annotation(
    annotation: &TypeAnnotation,
    substitution: &Substitution,
) -> TypeAnnotation {
    let kind = match &annotation.kind {
        TypeAnnotationKind::Named { name, arguments } => TypeAnnotationKind::Named {
            name: substitution
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone()),
            arguments: arguments
                .iter()
                .map(|argument| specialize_annotation(argument, substitution))
                .collect(),
        },
        TypeAnnotationKind::Union(types) => TypeAnnotationKind::Union(
            types
                .iter()
                .map(|ty| specialize_annotation(ty, substitution))
                .collect(),
        ),
    };
    TypeAnnotation {
        kind,
        span: annotation.span,
    }
}
