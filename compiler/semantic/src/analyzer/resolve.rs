use super::*;

pub(super) fn expression_class(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    match expression {
        Expr::Identifier(identifier) => scope
            .get(&identifier.name)
            .and_then(|binding| binding.class.clone())
            .or_else(|| {
                aliases
                    .get(&identifier.name)
                    .and_then(|value| value.strip_prefix("__class."))
                    .map(str::to_owned)
            }),
        Expr::Call(call) => match call.callee.as_ref() {
            Expr::Identifier(identifier) => aliases
                .get(&identifier.name)
                .and_then(|value| value.strip_prefix("__class."))
                .map(str::to_owned)
                .or_else(|| {
                    let function = aliases
                        .get(&identifier.name)
                        .map(String::as_str)
                        .unwrap_or(&identifier.name);
                    aliases
                        .get(&format!("__function_return_class.{function}"))
                        .cloned()
                }),
            Expr::Member(member) => {
                if let Expr::Identifier(module) = member.object.as_ref() {
                    let exported = format!("{}.{}", module.name, member.member.name);
                    if let Some(class) = aliases
                        .get(&format!("__module_class.{exported}"))
                        .cloned()
                        .or_else(|| {
                            let module = aliases
                                .get(&module.name)
                                .map(String::as_str)
                                .unwrap_or(&module.name);
                            aliases
                                .get(&format!(
                                    "__function_return_class.{module}.{}",
                                    member.member.name
                                ))
                                .cloned()
                        })
                    {
                        return Some(class);
                    }
                }
                expression_class(&member.object, scope, aliases).and_then(|class| {
                    aliases
                        .get(&format!(
                            "__class_method_return_class.{class}.{}",
                            member.member.name
                        ))
                        .cloned()
                })
            }
            _ => None,
        }
        .or_else(|| generated_object_call_class(call, scope, aliases)),
        Expr::Member(member) => {
            let class = expression_class(&member.object, scope, aliases)?;
            aliases
                .get(&format!(
                    "__class_field_class.{class}.{}",
                    member.member.name
                ))
                .cloned()
        }
        _ => None,
    }
}

pub(super) fn file_read_result_class(
    expression: &Expr,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    let Expr::Call(call) = expression else {
        return None;
    };
    let called = called_function_name(call.callee.as_ref(), aliases);
    let package_file_read = matches!(
        call.callee.as_ref(),
        Expr::Member(member)
            if member.member.name == "read"
                && matches!(
                    member.object.as_ref(),
                    Expr::Identifier(module)
                        if module.name == "file"
                            || aliases.get(&module.name).is_some_and(|name| name == "file")
                )
    );
    let literal_class = call.args.first().and_then(|argument| {
        let Expr::Literal(Literal::String { value, .. }) = &argument.value else {
            return None;
        };
        file_class_for_literal_path(value)
    });
    let local_file_read = called.as_deref() == Some("read")
        && (literal_class.is_some() || aliases.contains_key("__trait.File"));
    if called.as_deref() != Some("file.read") && !package_file_read && !local_file_read {
        return None;
    }
    Some(literal_class.unwrap_or("File").to_owned())
}

pub(super) fn file_read_receiver_type(
    expression: &Expr,
    aliases: &HashMap<String, String>,
) -> Option<ReceiverType> {
    let name = file_read_result_class(expression, aliases)?;
    let concrete = !aliases.contains_key(&format!("__trait.{name}"));
    let methods = aliases
        .get(&format!("__class_methods.{name}"))
        .filter(|methods| !methods.is_empty())
        .map(|methods| methods.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    Some(ReceiverType {
        name,
        concrete,
        methods,
    })
}

pub(super) fn called_function_name(
    callee: &Expr,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    match callee {
        Expr::Identifier(identifier) => Some(
            aliases
                .get(&identifier.name)
                .cloned()
                .unwrap_or_else(|| identifier.name.clone()),
        ),
        Expr::Member(member) => {
            let Expr::Identifier(module) = member.object.as_ref() else {
                return None;
            };
            let module = aliases
                .get(&module.name)
                .map(String::as_str)
                .unwrap_or(&module.name);
            Some(format!("{module}.{}", member.member.name))
        }
        _ => None,
    }
}

pub(super) fn file_class_for_literal_path(path: &str) -> Option<&'static str> {
    let extension = path.rsplit_once('.').map(|(_, extension)| extension)?;
    if extension.eq_ignore_ascii_case("wav") {
        Some("WAV")
    } else if extension.eq_ignore_ascii_case("csv") {
        Some("CSV")
    } else if extension.eq_ignore_ascii_case("json") {
        Some("Json")
    } else if extension.eq_ignore_ascii_case("yaml") || extension.eq_ignore_ascii_case("yml") {
        Some("Yaml")
    } else if extension.eq_ignore_ascii_case("mp3") {
        Some("MP3")
    } else if [
        "txt", "text", "md", "sev", "toml", "xml", "html", "log", "lua",
    ]
    .iter()
    .any(|known| extension.eq_ignore_ascii_case(known))
    {
        Some("Text")
    } else {
        None
    }
}

pub(super) fn refine_success_pattern_bindings(
    pattern: &MatchPattern,
    class: &str,
    scope: &mut HashMap<String, Binding>,
) {
    let MatchPattern::Constructor { name, fields } = pattern else {
        return;
    };
    if name != "ok" {
        return;
    }
    for field in fields {
        if let MatchPattern::Bind(name) = field {
            if let Some(binding) = scope.get_mut(&name.name) {
                binding.class = Some(class.to_owned());
            }
        }
    }
}

pub(super) fn success_pattern_receivers(
    pattern: &MatchPattern,
    receiver: &ReceiverType,
) -> BTreeMap<BindingId, ReceiverType> {
    let MatchPattern::Constructor { name, fields } = pattern else {
        return BTreeMap::new();
    };
    if name != "ok" {
        return BTreeMap::new();
    }
    fields
        .iter()
        .filter_map(|field| {
            let MatchPattern::Bind(name) = field else {
                return None;
            };
            Some((name.id, receiver.clone()))
        })
        .collect()
}

pub(super) fn imports_entire_module(module: &Module, module_name: &str) -> bool {
    module.items.iter().any(|item| {
        let Item::Import(import) = item else {
            return false;
        };
        match &import.kind {
            ImportKind::Local { path, .. } => {
                local_import_module_name(path).as_deref() == Some(module_name)
            }
            ImportKind::Module { path, .. } => {
                path.iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".")
                    == module_name
            }
            ImportKind::From { .. } => false,
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_class_function(
    id: FunctionId,
    class_name: &str,
    fields: &[String],
    name: &str,
    source_decorators: &[severian_ast::Decorator],
    source_params: &[severian_ast::Parameter],
    source_contract: Option<&severian_ast::FunctionContract>,
    body: &Block,
    source_tests: &[severian_ast::TestBlock],
    return_type: ValueType,
    global_scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Function, SemanticError> {
    let mut scope = global_scope.clone();
    for field in fields {
        scope.insert(
            field.clone(),
            Binding {
                reference: named_binding(field, format!("{class_name}.{field}")),
                ty: ValueType::Any,
                class: aliases
                    .get(&format!("__class_field_class.{class_name}.{field}"))
                    .cloned(),
                function_return: None,
                collection_len: None,
                mutable: true,
                field: true,
                integer_max: None,
                known_integer: None,
            },
        );
    }
    let mut params = Vec::new();
    for param in source_params {
        let ty = param
            .ty
            .as_ref()
            .map(lower_type)
            .transpose()?
            .unwrap_or(ValueType::Any);
        let default = param
            .default
            .as_ref()
            .map(|value| {
                lower_expression(value, &scope, signatures, aliases).map(|(value, _)| value)
            })
            .transpose()?;
        scope.insert(
            param.name.name.clone(),
            Binding {
                reference: source_binding(&param.name),
                ty,
                class: param.ty.as_ref().and_then(class_type_name),
                function_return: function_return_type(param.ty.as_ref()),
                collection_len: None,
                mutable: false,
                field: false,
                integer_max: None,
                known_integer: None,
            },
        );
        params.push(Parameter {
            name: scope[&param.name.name].reference.clone(),
            ty,
            default,
            receiver: param
                .ty
                .as_ref()
                .and_then(|ty| declared_receiver_type(ty, aliases)),
        });
    }
    let mut instructions = lower_block(body, &mut scope, return_type, signatures, aliases)?;
    if return_type != ValueType::Unit && !always_returns(&instructions) {
        return Err(error(
            body.span,
            format!("method `{class_name}.{name}` must return a value"),
        ));
    }
    let mut tests = Vec::new();
    for test in source_tests {
        let mut test_scope = global_scope.clone();
        add_test_bindings(&mut test_scope, &test.modes);
        let contract =
            lower_function_contract(test.contract.as_ref(), &test_scope, signatures, aliases)?;
        let mut test_instructions = lower_block(
            &test.body,
            &mut test_scope,
            ValueType::Unit,
            signatures,
            aliases,
        )?;
        if !test.modes.contains(&severian_ast::TestMode::Profile) {
            enforce_function_contract(&mut test_instructions, contract.as_ref());
        }
        tests.push(Test {
            name: test.name.as_ref().map(|name| name.name.clone()),
            modes: lower_test_modes(&test.modes),
            return_type: lower_test_return_type(test)?,
            contract,
            instructions: test_instructions,
        });
    }
    let contract = lower_function_contract(source_contract, &scope, signatures, aliases)?;
    enforce_function_contract(&mut instructions, contract.as_ref());
    Ok(Function {
        id,
        name: name.into(),
        native_symbol: None,
        decorators: decorator_metadata(source_decorators),
        contract,
        params,
        return_type,
        instructions,
        tests,
    })
}

pub(super) fn constructor_id(class: &str, name: &str, span: severian_ast::Span) -> FunctionId {
    FunctionId::from_name(&format!("{class}.{name}@{}:{}", span.start, span.end))
}

pub(super) fn lower_function_contract(
    contract: Option<&severian_ast::FunctionContract>,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Option<HirFunctionContract>, SemanticError> {
    contract
        .map(|contract| {
            let clauses = contract
                .clauses
                .iter()
                .map(|clause| {
                    let (condition, ty) =
                        lower_expression(&clause.condition, scope, signatures, aliases)?;
                    compatible(clause.condition.span(), ty, ValueType::Bool)?;
                    let mut dependencies = Vec::new();
                    collect_contract_dependencies(&condition, &mut dependencies);
                    dependencies.sort();
                    dependencies.dedup();
                    let dependency_types = dependencies
                        .iter()
                        .map(|name| {
                            scope
                                .get(&name.name)
                                .map_or(ValueType::Any, |binding| binding.ty)
                        })
                        .collect();
                    Ok(HirContractClause {
                        condition,
                        deferred: clause.deferred,
                        message: clause
                            .failure
                            .as_ref()
                            .map(|failure| failure.message.clone()),
                        location: clause
                            .failure
                            .as_ref()
                            .is_some_and(|failure| failure.location),
                        vars: clause.failure.as_ref().is_some_and(|failure| failure.vars),
                        dependencies,
                        dependency_types,
                    })
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            let capabilities = contract
                .capabilities
                .iter()
                .map(|capability| {
                    lower_expression(
                        &Expr::Identifier(capability.clone()),
                        scope,
                        signatures,
                        aliases,
                    )
                    .map(|(capability, _)| capability)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(HirFunctionContract {
                clauses,
                capabilities,
            })
        })
        .transpose()
}

pub(super) fn collect_imports(module: &Module) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for item in &module.items {
        let Item::Import(import) = item else { continue };
        match &import.kind {
            ImportKind::Local { path, alias } => {
                let Some(canonical) = local_import_module_name(path) else {
                    continue;
                };
                let exposed = alias
                    .as_ref()
                    .map(|alias| alias.name.clone())
                    .or_else(|| local_import_exposed_name(path));
                if let Some(exposed) = exposed {
                    aliases.insert(exposed, canonical);
                }
            }
            ImportKind::Module { path, alias } => {
                let canonical = path
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                let exposed = alias
                    .as_ref()
                    .unwrap_or_else(|| path.first().unwrap())
                    .name
                    .clone();
                aliases.insert(exposed, canonical);
            }
            ImportKind::From { module, names } => {
                let module = module
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                for name in names {
                    let exposed = name.alias.as_ref().unwrap_or(&name.name).name.clone();
                    aliases.insert(exposed, format!("{module}.{}", name.name.name));
                }
            }
        }
    }
    aliases
}

pub(super) fn aliases_with_decorators(
    aliases: &HashMap<String, String>,
    decorators: &[severian_ast::Decorator],
) -> HashMap<String, String> {
    let mut aliases = aliases.clone();
    for decorator in decorators {
        let package = decorator
            .name
            .segments
            .iter()
            .map(|segment| segment.name.as_str())
            .collect::<Vec<_>>()
            .join(".");
        for symbol in &decorator.symbols {
            aliases.insert(format!("__symbol.{}", symbol.spelling), package.clone());
            if let Some(function) =
                aliases.get(&format!("__symbol_alias.{package}.{}", symbol.spelling))
            {
                aliases.insert(symbol.spelling.clone(), function.clone());
            }
        }
    }
    aliases
}

pub(super) fn collect_imported_modules(module: &Module) -> HashSet<String> {
    module
        .items
        .iter()
        .filter_map(|item| {
            let Item::Import(import) = item else {
                return None;
            };
            match &import.kind {
                ImportKind::Local { path, alias } => alias
                    .as_ref()
                    .map(|alias| alias.name.clone())
                    .or_else(|| local_import_exposed_name(path)),
                ImportKind::Module { path, alias } => Some(
                    alias
                        .as_ref()
                        .unwrap_or_else(|| path.first().unwrap())
                        .name
                        .clone(),
                ),
                ImportKind::From { .. } => None,
            }
        })
        .collect()
}

pub(super) fn lower_decorators(
    decorators: &[severian_ast::Decorator],
    imported_modules: &HashSet<String>,
) -> Result<Vec<HirDecorator>, SemanticError> {
    let mut compile_policy_seen = false;
    for decorator in decorators {
        let root = &decorator.name.segments.first().unwrap().name;
        if root == "compile" {
            if compile_policy_seen {
                return Err(error(
                    decorator.span,
                    "a function may declare only one `@compile` backend policy",
                ));
            }
            compile_policy_seen = true;
            if decorator.name.segments.len() != 1
                || decorator.symbols.len() != 1
                || !matches!(
                    decorator.symbols[0].spelling.as_str(),
                    "auto" | "xla" | "triton"
                )
            {
                return Err(error(
                    decorator.span,
                    "`@compile` expects exactly one backend policy: `auto`, `xla`, or `triton`",
                ));
            }
        } else if !imported_modules.contains(root) {
            return Err(error(
                decorator.name.span,
                format!("decorator package `{root}` must be imported"),
            ));
        }
        let mut seen = HashSet::new();
        for symbol in &decorator.symbols {
            if !seen.insert(&symbol.spelling) {
                return Err(error(
                    symbol.span,
                    format!("duplicate decorator symbol `{}`", symbol.spelling),
                ));
            }
        }
    }
    Ok(decorator_metadata(decorators))
}

pub(super) fn decorator_metadata(decorators: &[severian_ast::Decorator]) -> Vec<HirDecorator> {
    decorators
        .iter()
        .map(|decorator| HirDecorator {
            package: decorator
                .name
                .segments
                .iter()
                .map(|segment| segment.name.as_str())
                .collect::<Vec<_>>()
                .join("."),
            symbols: decorator
                .symbols
                .iter()
                .map(|symbol| symbol.spelling.clone())
                .collect(),
        })
        .collect()
}
