use super::*;

#[derive(Clone)]
struct TraitEntry {
    canonical: String,
    namespace: Option<String>,
    aliases: HashMap<String, String>,
    declaration: severian_ast::TraitDecl,
}

pub(super) fn expand_trait_compositions(
    module: &Module,
    interfaces: &[PackageInterface],
) -> Result<(Module, Vec<PackageInterface>), SemanticError> {
    let mut registry = HashMap::<String, TraitEntry>::new();
    let local_aliases = collect_imports(module);
    for item in &module.items {
        let Item::Trait(declaration) = item else {
            continue;
        };
        let canonical = declaration.name.name.clone();
        registry.insert(
            canonical.clone(),
            TraitEntry {
                canonical,
                namespace: None,
                aliases: local_aliases.clone(),
                declaration: declaration.clone(),
            },
        );
    }
    for interface in interfaces {
        let aliases = collect_imports(&interface.module);
        for item in &interface.module.items {
            let Item::Trait(declaration) = item else {
                continue;
            };
            let canonical = format!("{}.{}", interface.name, declaration.name.name);
            let entry = TraitEntry {
                canonical: canonical.clone(),
                namespace: Some(interface.name.clone()),
                aliases: aliases.clone(),
                declaration: declaration.clone(),
            };
            registry.insert(canonical.clone(), entry.clone());
            if let Some(package) = &interface.export_package {
                registry.insert(format!("{package}.{}", declaration.name.name), entry);
            }
        }
    }

    let canonical_keys = registry
        .values()
        .map(|entry| entry.canonical.clone())
        .collect::<HashSet<_>>();
    let mut cache = HashMap::<String, severian_ast::TraitDecl>::new();
    for key in canonical_keys {
        expand_trait(&key, &registry, &mut cache, &mut Vec::new())?;
    }

    let mut expanded_module = module.clone();
    for item in &mut expanded_module.items {
        let Item::Trait(declaration) = item else {
            continue;
        };
        *declaration = cache[&declaration.name.name].clone();
    }
    let mut expanded_interfaces = interfaces.to_vec();
    for interface in &mut expanded_interfaces {
        for item in &mut interface.module.items {
            let Item::Trait(declaration) = item else {
                continue;
            };
            let key = format!("{}.{}", interface.name, declaration.name.name);
            *declaration = cache[&key].clone();
        }
    }
    Ok((expanded_module, expanded_interfaces))
}

fn expand_trait(
    key: &str,
    registry: &HashMap<String, TraitEntry>,
    cache: &mut HashMap<String, severian_ast::TraitDecl>,
    active: &mut Vec<String>,
) -> Result<severian_ast::TraitDecl, SemanticError> {
    if let Some(expanded) = cache.get(key) {
        return Ok(expanded.clone());
    }
    if let Some(index) = active.iter().position(|candidate| candidate == key) {
        let mut cycle = active[index..].to_vec();
        cycle.push(key.to_owned());
        let declaration = &registry[key].declaration;
        return Err(error(
            declaration.name.span,
            format!("trait composition cycle: {}", cycle.join(" -> ")),
        ));
    }
    let entry = registry
        .get(key)
        .expect("trait expansion keys come from the registry");
    active.push(key.to_owned());
    let mut expanded = entry.declaration.clone();
    let mut method_signatures = expanded
        .methods
        .iter()
        .map(|method| (method.name.name.clone(), trait_method_signature(method)))
        .collect::<HashMap<_, _>>();
    let mut operator_signatures = expanded
        .operators
        .iter()
        .map(|operator| {
            (
                (operator.symbol.clone(), operator.params.len()),
                trait_operator_signature(operator),
            )
        })
        .collect::<HashMap<_, _>>();

    for composed in &entry.declaration.composed_traits {
        let raw = declaration_type_name(composed).ok_or_else(|| {
            error(
                composed.span(),
                "a composed trait requirement must be a named trait",
            )
        })?;
        let composed_key = resolve_composed_trait(&raw, entry, registry).ok_or_else(|| {
            error(
                composed.span(),
                format!(
                    "unknown trait `{raw}` composed by `{}`",
                    entry.declaration.name.name
                ),
            )
        })?;
        let inherited = expand_trait(&composed_key, registry, cache, active)?;
        let substitutions = composition_substitutions(&inherited, composed)?;
        for method in &inherited.methods {
            let method = substitute_trait_method(method, &substitutions);
            let signature = trait_method_signature(&method);
            match method_signatures.get(&method.name.name) {
                Some(existing) if existing == &signature => continue,
                Some(existing) => {
                    return Err(error(
                        composed.span(),
                        format!(
                            "trait `{}` composes conflicting requirements for `{}`: `{existing}` and `{signature}`",
                            entry.declaration.name.name, method.name.name
                        ),
                    ))
                }
                None => {
                    method_signatures.insert(method.name.name.clone(), signature);
                    expanded.methods.push(method);
                }
            }
        }
        for operator in &inherited.operators {
            let operator = substitute_trait_operator(operator, &substitutions);
            let identity = (operator.symbol.clone(), operator.params.len());
            let signature = trait_operator_signature(&operator);
            match operator_signatures.get(&identity) {
                Some(existing) if existing == &signature => continue,
                Some(existing) => {
                    return Err(error(
                        composed.span(),
                        format!(
                            "trait `{}` composes conflicting requirements for operator `{}`: `{existing}` and `{signature}`",
                            entry.declaration.name.name, operator.symbol
                        ),
                    ))
                }
                None => {
                    operator_signatures.insert(identity, signature);
                    expanded.operators.push(operator);
                }
            }
        }
    }
    active.pop();
    cache.insert(key.to_owned(), expanded.clone());
    Ok(expanded)
}

fn resolve_composed_trait(
    raw: &str,
    owner: &TraitEntry,
    registry: &HashMap<String, TraitEntry>,
) -> Option<String> {
    if !raw.contains('.') {
        if let Some(namespace) = &owner.namespace {
            if let Some(entry) = registry.get(&format!("{namespace}.{raw}")) {
                return Some(entry.canonical.clone());
            }
        } else if let Some(entry) = registry.get(raw) {
            return Some(entry.canonical.clone());
        }
        if let Some(canonical) = owner.aliases.get(raw) {
            if let Some(entry) = registry.get(canonical) {
                return Some(entry.canonical.clone());
            }
        }
        return registry.get(raw).map(|entry| entry.canonical.clone());
    }
    let canonical = canonical_declared_type_name(raw, &owner.aliases);
    if let Some(entry) = registry.get(&canonical) {
        return Some(entry.canonical.clone());
    }
    registry.get(raw).map(|entry| entry.canonical.clone())
}

fn composition_substitutions(
    declaration: &severian_ast::TraitDecl,
    composed: &Type,
) -> Result<HashMap<String, Type>, SemanticError> {
    let Type::Named(path) = composed else {
        unreachable!()
    };
    if path.args.len() != declaration.generic_params.len() {
        return Err(error(
            composed.span(),
            format!(
                "trait `{}` expects {} type argument(s), received {}",
                declaration.name.name,
                declaration.generic_params.len(),
                path.args.len()
            ),
        ));
    }
    declaration
        .generic_params
        .iter()
        .zip(&path.args)
        .map(|(parameter, argument)| {
            let TypeArg::Type { ty, .. } = argument else {
                return Err(error(
                    argument.span(),
                    format!("trait `{}` requires type arguments", declaration.name.name),
                ));
            };
            Ok((parameter.name.name.clone(), ty.as_ref().clone()))
        })
        .collect()
}

fn substitute_trait_method(
    method: &severian_ast::TraitMethod,
    substitutions: &HashMap<String, Type>,
) -> severian_ast::TraitMethod {
    severian_ast::TraitMethod {
        params: method
            .params
            .iter()
            .map(|parameter| severian_ast::Parameter {
                ty: parameter
                    .ty
                    .as_ref()
                    .map(|ty| substitute_declared_type(ty, substitutions)),
                ..parameter.clone()
            })
            .collect(),
        return_type: method
            .return_type
            .as_ref()
            .map(|ty| substitute_declared_type(ty, substitutions)),
        ..method.clone()
    }
}

fn substitute_trait_operator(
    operator: &severian_ast::TraitOperator,
    substitutions: &HashMap<String, Type>,
) -> severian_ast::TraitOperator {
    severian_ast::TraitOperator {
        params: operator
            .params
            .iter()
            .map(|parameter| severian_ast::Parameter {
                ty: parameter
                    .ty
                    .as_ref()
                    .map(|ty| substitute_declared_type(ty, substitutions)),
                ..parameter.clone()
            })
            .collect(),
        return_type: operator
            .return_type
            .as_ref()
            .map(|ty| substitute_declared_type(ty, substitutions)),
        ..operator.clone()
    }
}

pub(super) fn trait_operator_signature(operator: &severian_ast::TraitOperator) -> String {
    let params = operator
        .params
        .iter()
        .map(|parameter| {
            parameter
                .ty
                .as_ref()
                .map(declaration_type_key)
                .unwrap_or_else(|| "Any".into())
        })
        .collect::<Vec<_>>()
        .join(", ");
    let returns = operator
        .return_type
        .as_ref()
        .map(declaration_type_key)
        .unwrap_or_else(|| "unit".into());
    format!("operator {}({params}) -> {returns}", operator.symbol)
}

pub(super) fn validate_trait_implementations(
    module: &Module,
    interfaces: &[PackageInterface],
    aliases: &HashMap<String, String>,
) -> Result<(), SemanticError> {
    let mut traits = HashMap::<String, &severian_ast::TraitDecl>::new();
    for item in &module.items {
        if let Item::Trait(declaration) = item {
            traits.insert(declaration.name.name.clone(), declaration);
        }
    }
    for interface in interfaces {
        for item in &interface.module.items {
            if let Item::Trait(declaration) = item {
                traits.insert(
                    format!("{}.{}", interface.name, declaration.name.name),
                    declaration,
                );
                if let Some(package) = &interface.export_package {
                    traits.insert(format!("{package}.{}", declaration.name.name), declaration);
                }
            }
        }
    }

    for item in &module.items {
        let Item::Class(class) = item else { continue };
        for implemented in &class.traits {
            let raw = declaration_type_name(implemented).ok_or_else(|| {
                error(
                    implemented.span(),
                    "trait implementation requires a named trait",
                )
            })?;
            if raw.rsplit('.').next() == Some("From") {
                validate_from_implementation(class, implemented)?;
                continue;
            }
            let canonical = canonical_declared_type_name(&raw, aliases);
            let declaration = traits
                .get(&canonical)
                .or_else(|| traits.get(&raw))
                .ok_or_else(|| {
                    error(
                        implemented.span(),
                        format!("unknown trait `{raw}` implemented by `{}`", class.name.name),
                    )
                })?;
            let substitutions = trait_substitutions(declaration, implemented, class)?;
            for required in &declaration.methods {
                let Some(method) = class
                    .methods
                    .iter()
                    .find(|method| method.name.name == required.name.name)
                else {
                    return Err(error(
                        implemented.span(),
                        format!(
                            "class `{}` does not implement `{}` required by trait `{raw}`",
                            class.name.name, required.name.name
                        ),
                    ));
                };
                let parameters_match = method.params.len() == required.params.len()
                    && method
                        .params
                        .iter()
                        .zip(&required.params)
                        .all(|(actual, expected)| {
                            let expected = expected
                                .ty
                                .as_ref()
                                .map(|ty| substitute_declared_type(ty, &substitutions));
                            optional_declaration_types_match(actual.ty.as_ref(), expected.as_ref())
                        });
                let expected_return = required
                    .return_type
                    .as_ref()
                    .map(|ty| substitute_declared_type(ty, &substitutions));
                let return_matches = optional_declaration_types_match(
                    method.return_type.as_ref(),
                    expected_return.as_ref(),
                );
                if !parameters_match || !return_matches {
                    return Err(error(
                        method.name.span,
                        format!(
                            "method `{}.{}` does not match trait `{raw}`; expected `{}`",
                            class.name.name,
                            method.name.name,
                            trait_method_signature(required)
                        ),
                    ));
                }
            }
            for required in &declaration.operators {
                let required = substitute_trait_operator(required, &substitutions);
                if !builtin_operator_satisfies(&required) {
                    return Err(error(
                        implemented.span(),
                        format!(
                            "class `{}` does not satisfy `{}` required by trait `{raw}`",
                            class.name.name,
                            trait_operator_signature(&required)
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn builtin_operator_satisfies(operator: &severian_ast::TraitOperator) -> bool {
    let parameters = operator
        .params
        .iter()
        .map(|parameter| parameter.ty.as_ref().map(declaration_type_key))
        .collect::<Option<Vec<_>>>();
    let returns = operator.return_type.as_ref().map(declaration_type_key);
    let (Some(parameters), Some(returns)) = (parameters, returns) else {
        return false;
    };
    match (operator.symbol.as_str(), parameters.as_slice()) {
        ("|" | "&" | "^", [left, right]) => {
            integer_type_name(left) && right == left && returns == *left
        }
        ("and" | "or", [left, right]) => left == "bool" && right == left && returns == *left,
        ("not", [value]) => value == "bool" && returns == *value,
        ("+" | "-" | "*" | "/", [left, right]) => {
            left == right
                && returns == *left
                && (integer_type_name(left)
                    || matches!(left.as_str(), "float" | "f32" | "f64")
                    || left.starts_with("Tensor["))
        }
        _ => false,
    }
}

fn integer_type_name(name: &str) -> bool {
    matches!(
        name,
        "int" | "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
    )
}

fn trait_substitutions(
    declaration: &severian_ast::TraitDecl,
    implemented: &Type,
    class: &severian_ast::ClassDecl,
) -> Result<HashMap<String, Type>, SemanticError> {
    let Type::Named(path) = implemented else {
        unreachable!()
    };
    if path.args.len() != declaration.generic_params.len() {
        return Err(error(
            implemented.span(),
            format!(
                "trait `{}` expects {} type argument(s), received {}",
                declaration.name.name,
                declaration.generic_params.len(),
                path.args.len()
            ),
        ));
    }
    declaration
        .generic_params
        .iter()
        .zip(&path.args)
        .map(|(parameter, argument)| {
            let TypeArg::Type { ty, .. } = argument else {
                return Err(error(
                    argument.span(),
                    format!("trait `{}` requires type arguments", declaration.name.name),
                ));
            };
            Ok((parameter.name.name.clone(), ty.as_ref().clone()))
        })
        .chain(std::iter::once(Ok((
            "Self".into(),
            Type::Named(severian_ast::TypePath {
                span: class.name.span,
                segments: vec![class.name.clone()],
                args: Vec::new(),
            }),
        ))))
        .collect()
}

fn validate_from_implementation(
    class: &severian_ast::ClassDecl,
    implemented: &Type,
) -> Result<(), SemanticError> {
    let method = class
        .methods
        .iter()
        .find(|method| method.name.name == "from")
        .ok_or_else(|| {
            error(
                implemented.span(),
                format!(
                    "class `{}` implements `From` but does not define `from(value)`",
                    class.name.name
                ),
            )
        })?;
    if method.params.len() != 1 {
        return Err(error(
            method.name.span,
            "a `From` implementation must accept exactly one source value",
        ));
    }
    let Type::Named(path) = implemented else {
        unreachable!()
    };
    if let Some(expected) = path.args.first().and_then(TypeArg::as_type) {
        if !optional_declaration_types_match(method.params[0].ty.as_ref(), Some(expected)) {
            return Err(error(
                method.params[0].span,
                "the `from` parameter must match the source type in `From[Source]`",
            ));
        }
    }
    if method
        .return_type
        .as_ref()
        .and_then(class_type_name)
        .as_deref()
        != Some(class.name.name.as_str())
    {
        return Err(error(
            method.name.span,
            format!(
                "`{}.from` must return `{}`",
                class.name.name, class.name.name
            ),
        ));
    }
    Ok(())
}

pub(super) fn declaration_type_name(ty: &Type) -> Option<String> {
    let Type::Named(path) = ty else { return None };
    Some(
        path.segments
            .iter()
            .map(|segment| segment.name.as_str())
            .collect::<Vec<_>>()
            .join("."),
    )
}

pub(super) fn canonical_declared_type_name(
    name: &str,
    aliases: &HashMap<String, String>,
) -> String {
    let (head, tail) = name.split_once('.').unwrap_or((name, ""));
    let Some(canonical) = aliases.get(head) else {
        return name.to_owned();
    };
    if tail.is_empty() || canonical.ends_with(&format!(".{tail}")) {
        canonical.clone()
    } else {
        format!("{canonical}.{tail}")
    }
}

pub(super) fn optional_declaration_types_match(left: Option<&Type>, right: Option<&Type>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => declaration_type_key(left) == declaration_type_key(right),
        (None, Some(right)) | (Some(right), None) => declaration_type_key(right) == "unit",
    }
}

pub(super) fn declaration_type_key(ty: &Type) -> String {
    match ty {
        Type::Named(path) => {
            let mut result = path
                .segments
                .last()
                .map(|segment| segment.name.clone())
                .unwrap_or_default();
            if !path.args.is_empty() {
                let arguments = path
                    .args
                    .iter()
                    .map(|argument| match argument {
                        TypeArg::Type { ty, .. } => declaration_type_key(ty),
                        TypeArg::Dimension { size, .. } => size.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                result.push('[');
                result.push_str(&arguments);
                result.push(']');
            }
            result
        }
        Type::List { element, .. } => format!("list[{}]", declaration_type_key(element)),
        Type::Tuple { elements, .. } => format!(
            "tuple[{}]",
            elements
                .iter()
                .map(declaration_type_key)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Union { alternatives, .. } => alternatives
            .iter()
            .map(declaration_type_key)
            .collect::<Vec<_>>()
            .join(" | "),
        Type::Map { key, value, .. } => format!(
            "map[{}, {}]",
            declaration_type_key(key),
            declaration_type_key(value)
        ),
        Type::Set { element, .. } => format!("set[{}]", declaration_type_key(element)),
        Type::Result { ok, err, .. } => format!(
            "Result[{}, {}]",
            declaration_type_key(ok),
            declaration_type_key(err)
        ),
        Type::Option { some, .. } => format!("Option[{}]", declaration_type_key(some)),
        Type::Function {
            params, returns, ..
        } => {
            let mut parts = params.iter().map(declaration_type_key).collect::<Vec<_>>();
            parts.push(declaration_type_key(returns));
            format!("Function[{}]", parts.join(", "))
        }
        Type::Future { output, .. } => format!("Future[{}]", declaration_type_key(output)),
        Type::Reference { mutable, inner, .. } => format!(
            "{}{}",
            if *mutable { "mut &" } else { "&" },
            declaration_type_key(inner)
        ),
    }
}

pub(super) fn trait_method_signature(method: &severian_ast::TraitMethod) -> String {
    let params = method
        .params
        .iter()
        .map(|parameter| {
            parameter
                .ty
                .as_ref()
                .map(declaration_type_key)
                .unwrap_or_else(|| "Any".into())
        })
        .collect::<Vec<_>>()
        .join(", ");
    let returns = method
        .return_type
        .as_ref()
        .map(declaration_type_key)
        .unwrap_or_else(|| "unit".into());
    format!("{}({params}) -> {returns}", method.name.name)
}
