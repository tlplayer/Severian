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
            })
            .or_else(|| {
                aliases
                    .get(&format!("__enum_variant_owner.{}", identifier.name))
                    .cloned()
            }),
        Expr::Call(call) => match call.callee.as_ref() {
            Expr::Identifier(identifier) => aliases
                .get(&identifier.name)
                .and_then(|value| value.strip_prefix("__class."))
                .map(str::to_owned)
                .or_else(|| {
                    aliases
                        .get(&format!("__enum_variant_owner.{}", identifier.name))
                        .cloned()
                })
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

pub(super) fn expression_enum_variant(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    match expression {
        Expr::Identifier(identifier) => scope
            .get(&identifier.name)
            .and_then(|binding| binding.enum_variant.clone())
            .or_else(|| {
                aliases
                    .contains_key(&format!("__enum_variant_owner.{}", identifier.name))
                    .then(|| identifier.name.clone())
            }),
        Expr::Call(call) => match call.callee.as_ref() {
            Expr::Identifier(identifier)
                if aliases.contains_key(&format!("__enum_variant_owner.{}", identifier.name)) =>
            {
                Some(identifier.name.clone())
            }
            _ => None,
        },
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
        file_class_for_literal_path(value, aliases)
    });
    let local_file_read = called.as_deref() == Some("read")
        && (literal_class.is_some() || aliases.contains_key("__trait.File"));
    if called.as_deref() != Some("file.read") && !package_file_read && !local_file_read {
        return None;
    }
    let class = literal_class.as_deref().unwrap_or("File");
    aliases
        .get(&format!("__module_class.file.{class}"))
        .cloned()
        .or_else(|| {
            let suffix = format!(".{class}");
            let mut matches = aliases
                .iter()
                .filter(|(key, _)| key.starts_with("__module_class.") && key.ends_with(&suffix))
                .map(|(_, value)| value)
                .collect::<HashSet<_>>();
            (matches.len() == 1).then(|| matches.drain().next().unwrap().clone())
        })
        .or_else(|| Some(class.to_owned()))
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

pub(super) fn file_class_for_literal_path(
    path: &str,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    let extension = format!(
        ".{}",
        path.rsplit_once('.').map(|(_, extension)| extension)?
    )
    .to_lowercase();
    let suffix = format!(".extensions.{extension}");
    let mut providers = aliases
        .iter()
        .filter_map(|(key, provider)| {
            let registry = key
                .strip_prefix("__trait_registry_match.")?
                .strip_suffix(&suffix)?;
            registry
                .rsplit_once('.')
                .map_or(registry, |(_, name)| name)
                .eq("Reader")
                .then(|| provider.clone())
        })
        .collect::<Vec<_>>();
    providers.sort();
    providers.dedup();
    let [provider] = providers.as_slice() else {
        return None;
    };
    aliases
        .get(&format!(
            "__trait_registry_provider_property.{provider}.document_class"
        ))
        .cloned()
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
    decorators: Vec<HirDecorator>,
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
    scope.insert(
        "self".into(),
        Binding {
            reference: named_binding("self", format!("{class_name}.self")),
            ty: ValueType::Any,
            class: Some(class_name.into()),
            enum_variant: None,
            function_return: None,
            collection_len: None,
            mutable: false,
            field: false,
            integer_max: None,
            known_integer: None,
            any_origin: Some(AnyOrigin::Explicit),
        },
    );
    for field in fields {
        let ty = aliases
            .get(&format!("__class_field_type.{class_name}.{field}"))
            .and_then(|ty| decode_field_type(ty))
            .unwrap_or(ValueType::Any);
        scope.insert(
            field.clone(),
            Binding {
                reference: named_binding(field, format!("{class_name}.{field}")),
                ty,
                class: aliases
                    .get(&format!("__class_field_class.{class_name}.{field}"))
                    .cloned(),
                enum_variant: None,
                function_return: None,
                collection_len: None,
                mutable: true,
                field: true,
                integer_max: None,
                known_integer: None,
                any_origin: matches!(ty, ValueType::Any | ValueType::TensorAny).then_some(
                    if aliases.contains_key(&format!("__class_field_type.{class_name}.{field}")) {
                        AnyOrigin::Explicit
                    } else {
                        AnyOrigin::InferenceFallback
                    },
                ),
            },
        );
    }
    let mut params = Vec::new();
    for param in source_params {
        let ty = param
            .ty
            .as_ref()
            .map(|ty| declared_value_type(ty, aliases))
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
                class: param
                    .ty
                    .as_ref()
                    .and_then(|ty| resolved_class_type_name(ty, aliases)),
                enum_variant: None,
                function_return: function_return_type(param.ty.as_ref()),
                collection_len: None,
                mutable: false,
                field: false,
                integer_max: None,
                known_integer: None,
                any_origin: declared_any_origin(param.ty.as_ref(), ty),
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
    wrap_scoped_behaviors(&mut instructions, &decorators);
    Ok(Function {
        id,
        name: name.into(),
        native_symbol: None,
        decorators,
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

pub(super) fn function_semantic_decorators(
    decorators: &[severian_ast::Decorator],
    contract: Option<&severian_ast::FunctionContract>,
    trait_semantics: &TraitSemantics,
) -> Result<
    (
        Vec<severian_ast::Decorator>,
        Option<severian_ast::FunctionContract>,
    ),
    SemanticError,
> {
    let mut decorators = decorators.to_vec();
    let Some(contract) = contract else {
        return Ok((decorators, None));
    };
    let mut filtered = contract.clone();
    filtered.clauses.clear();
    for clause in &contract.clauses {
        let Some(decorator) =
            semantic_decorator_from_expression(&clause.condition, trait_semantics)?
        else {
            filtered.clauses.push(clause.clone());
            continue;
        };
        if clause.deferred || clause.failure.is_some() {
            return Err(error(
                clause.span,
                "a scoped semantic behavior cannot be deferred or have a contract failure action",
            ));
        }
        decorators.push(decorator);
    }
    Ok((decorators, Some(filtered)))
}

fn semantic_decorator_from_expression(
    expression: &Expr,
    trait_semantics: &TraitSemantics,
) -> Result<Option<severian_ast::Decorator>, SemanticError> {
    let (segments, arguments, span) = match expression {
        Expr::Identifier(identifier) => (vec![identifier.clone()], &[][..], identifier.span),
        Expr::Call(call) => {
            let Expr::Identifier(identifier) = call.callee.as_ref() else {
                return Ok(None);
            };
            (vec![identifier.clone()], call.args.as_slice(), call.span)
        }
        _ => return Ok(None),
    };
    let name = segments
        .iter()
        .map(|segment| segment.name.as_str())
        .collect::<Vec<_>>()
        .join(".");
    if !trait_semantics.decorators.contains_key(&name) {
        return Ok(None);
    }
    let mut symbols = Vec::new();
    for argument in arguments {
        let Expr::Identifier(value) = &argument.value else {
            return Err(error(
                argument.value.span(),
                format!(
                    "semantic behavior `{name}` accepts only trait selectors or named identifier policies"
                ),
            ));
        };
        let (spelling, value) = if let Some(parameter) = &argument.name {
            (parameter.name.clone(), Some(value.name.clone()))
        } else {
            (value.name.clone(), None)
        };
        symbols.push(severian_ast::DecoratorSymbol {
            span: argument.span,
            spelling,
            value,
        });
    }
    Ok(Some(severian_ast::Decorator {
        span,
        name: severian_ast::TypePath {
            span: Span::new(
                segments.first().unwrap().span.start,
                segments.last().unwrap().span.end,
            ),
            segments,
            args: Vec::new(),
        },
        symbols,
    }))
}

pub(super) fn wrap_scoped_behaviors(
    instructions: &mut Vec<Instruction>,
    decorators: &[HirDecorator],
) {
    let mut scoped_behaviors = Vec::new();
    for behavior in decorators
        .iter()
        .filter_map(|decorator| decorator.semantic_context.as_ref())
        .flat_map(|context| context.scoped_behaviors.iter())
    {
        if !scoped_behaviors
            .iter()
            .any(|existing: &HirScopedBehavior| existing.provider == behavior.provider)
        {
            scoped_behaviors.push(behavior.clone());
        }
    }
    if scoped_behaviors.is_empty() {
        return;
    }
    let body = std::mem::take(instructions);
    instructions.push(Instruction::With {
        placement: TaskPlacement::Default,
        resources: Vec::new(),
        scoped_behaviors,
        instructions: body,
    });
}

pub(super) fn aliases_with_decorators(
    aliases: &HashMap<String, String>,
    decorators: &[severian_ast::Decorator],
    trait_semantics: &TraitSemantics,
) -> Result<HashMap<String, String>, SemanticError> {
    let mut aliases = aliases.clone();
    for decorator in decorators {
        let package = decorator
            .name
            .segments
            .iter()
            .map(|segment| segment.name.as_str())
            .collect::<Vec<_>>()
            .join(".");
        aliases.insert(format!("__capability.{package}"), String::new());
        for symbol in &decorator.symbols {
            if symbol.value.is_some() {
                continue;
            }
            aliases.insert(format!("__symbol.{}", symbol.spelling), package.clone());
            if let Some(function) =
                aliases.get(&format!("__symbol_alias.{package}.{}", symbol.spelling))
            {
                aliases.insert(symbol.spelling.clone(), function.clone());
            }
        }
        if let Some(context) = semantic_context_for(decorator, trait_semantics)? {
            for operator in context.operators {
                aliases.insert(
                    format!("__semantic.operator_candidates.{}", operator.name),
                    operator.candidates.join(","),
                );
                if let Some(selected) = operator.selected {
                    aliases.insert(format!("__semantic.operator.{}", operator.name), selected);
                }
            }
            for operation in context.operations {
                aliases.insert(
                    format!("__semantic.operation_candidates.{}", operation.name),
                    operation.candidates.join(","),
                );
                if let Some(selected) = operation.selected {
                    aliases.insert(format!("__semantic.operation.{}", operation.name), selected);
                }
            }
        }
    }
    Ok(aliases)
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
    trait_semantics: &TraitSemantics,
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
        } else if !imported_modules.contains(root)
            && !trait_semantics
                .decorators
                .contains_key(&decorator_name(decorator))
        {
            return Err(error(
                decorator.name.span,
                format!("decorator package `{root}` must be imported"),
            ));
        }
        if root == "bits" {
            if decorator.symbols.is_empty() {
                return Err(error(
                    decorator.span,
                    "`@bits` requires an explicit non-empty subset of `|`, `&`, and `^`",
                ));
            }
            for symbol in &decorator.symbols {
                if !matches!(symbol.spelling.as_str(), "|" | "&" | "^") {
                    return Err(error(
                        symbol.span,
                        format!(
                            "unknown `bits` capability member `{}`; expected `|`, `&`, or `^`",
                            symbol.spelling
                        ),
                    ));
                }
            }
        }
        let mut seen = HashSet::new();
        for symbol in &decorator.symbols {
            let identity = (&symbol.spelling, &symbol.value);
            if !seen.insert(identity) {
                return Err(error(
                    symbol.span,
                    format!("duplicate decorator symbol `{}`", symbol.spelling),
                ));
            }
        }
        semantic_context_for(decorator, trait_semantics)?;
    }
    decorator_metadata(decorators, trait_semantics)
}

pub(super) fn decorator_metadata(
    decorators: &[severian_ast::Decorator],
    trait_semantics: &TraitSemantics,
) -> Result<Vec<HirDecorator>, SemanticError> {
    decorators
        .iter()
        .map(|decorator| {
            Ok(HirDecorator {
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
                    .filter(|symbol| symbol.value.is_none())
                    .map(|symbol| symbol.spelling.clone())
                    .collect(),
                options: decorator
                    .symbols
                    .iter()
                    .filter_map(|symbol| {
                        symbol.value.as_ref().map(|value| HirDecoratorOption {
                            name: symbol.spelling.clone(),
                            value: value.clone(),
                        })
                    })
                    .collect(),
                semantic_context: semantic_context_for(decorator, trait_semantics)?,
            })
        })
        .collect()
}

fn semantic_context_for(
    decorator: &severian_ast::Decorator,
    trait_semantics: &TraitSemantics,
) -> Result<Option<SemanticContext>, SemanticError> {
    let decorator_name = decorator_name(decorator);
    let Some(definition) = trait_semantics.decorators.get(&decorator_name) else {
        return Ok(None);
    };
    let namespace = &trait_semantics.namespaces[&definition.owner];
    let mut selected_traits = Vec::new();
    for selector in decorator
        .symbols
        .iter()
        .filter(|symbol| symbol.value.is_none() && symbol.spelling != "auto")
    {
        let Some(selected) = trait_semantics.decorators.get(&selector.spelling) else {
            return Err(error(
                selector.span,
                format!(
                    "unknown selector `{}` for semantic decorator `@{decorator_name}`",
                    selector.spelling
                ),
            ));
        };
        if !namespace.traits.contains(&selected.owner) {
            return Err(error(
                selector.span,
                format!(
                    "selector `@{}` is not part of `{}`'s composed trait namespace",
                    selector.spelling, definition.owner
                ),
            ));
        }
        if selected.owner != definition.owner && !selected_traits.contains(&selected.owner) {
            selected_traits.push(selected.owner.clone());
        }
    }

    let mut policies = definition.policies.clone();
    for option in decorator
        .symbols
        .iter()
        .filter_map(|symbol| symbol.value.as_ref().map(|value| (&symbol.spelling, value)))
    {
        let Some(existing) = policies.iter_mut().find(|policy| policy.0 == *option.0) else {
            let symbol = decorator
                .symbols
                .iter()
                .find(|symbol| symbol.spelling == *option.0 && symbol.value.is_some())
                .expect("the named policy came from this decorator");
            return Err(error(
                symbol.span,
                format!(
                    "unknown policy `{}` for semantic decorator `@{decorator_name}`",
                    option.0
                ),
            ));
        };
        existing.1 = option.1.clone();
    }

    let active_traits = if selected_traits.is_empty() {
        namespace.traits.clone()
    } else {
        let mut traits = vec![definition.owner.clone()];
        traits.extend(selected_traits.iter().cloned());
        traits
    };
    Ok(Some(SemanticContext {
        capability: definition.owner.clone(),
        traits: active_traits,
        operators: semantic_members(&namespace.operators, &selected_traits),
        operations: semantic_members(&namespace.operations, &selected_traits),
        scoped_behaviors: semantic_scoped_behaviors(namespace, &definition.owner, &selected_traits),
        policies: policies
            .into_iter()
            .map(|(name, value)| HirDecoratorOption { name, value })
            .collect(),
    }))
}

fn semantic_scoped_behaviors(
    namespace: &TraitSemanticNamespace,
    owner: &str,
    selected_traits: &[String],
) -> Vec<HirScopedBehavior> {
    namespace
        .scoped_behaviors
        .iter()
        .filter(|behavior| {
            selected_traits.is_empty()
                || behavior.trait_name == owner
                || selected_traits.contains(&behavior.trait_name)
        })
        .map(|behavior| HirScopedBehavior {
            provider: behavior.trait_name.clone(),
            with_member: format!("{}::with", behavior.trait_name),
            without_member: format!("{}::without", behavior.trait_name),
        })
        .collect()
}

fn semantic_members(
    members: &BTreeMap<String, Vec<TraitMemberProvider>>,
    selected_traits: &[String],
) -> Vec<SemanticMember> {
    members
        .iter()
        .map(|(name, providers)| {
            let matching_providers = providers
                .iter()
                .filter(|provider| selected_traits.contains(&provider.trait_name))
                .collect::<Vec<_>>();
            let resolution_candidates = if matching_providers.is_empty() {
                providers.iter().collect::<Vec<_>>()
            } else {
                matching_providers
            };
            SemanticMember {
                name: name.clone(),
                candidates: providers
                    .iter()
                    .map(|provider| provider.qualified_member.clone())
                    .collect(),
                selected: (resolution_candidates.len() == 1)
                    .then(|| resolution_candidates[0].qualified_member.clone()),
            }
        })
        .collect()
}

fn decorator_name(decorator: &severian_ast::Decorator) -> String {
    decorator
        .name
        .segments
        .iter()
        .map(|segment| segment.name.as_str())
        .collect::<Vec<_>>()
        .join(".")
}
