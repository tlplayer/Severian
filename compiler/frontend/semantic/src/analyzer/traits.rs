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
) -> Result<(Module, Vec<PackageInterface>, TraitSemantics), SemanticError> {
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
    let semantics = build_trait_semantics(&registry)?;

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
    Ok((expanded_module, expanded_interfaces, semantics))
}

fn build_trait_semantics(
    registry: &HashMap<String, TraitEntry>,
) -> Result<TraitSemantics, SemanticError> {
    let mut semantics = TraitSemantics::default();
    let mut keys = registry
        .values()
        .map(|entry| entry.canonical.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    keys.sort();

    for key in &keys {
        let entry = &registry[key];
        for decorator in &entry.declaration.decorators {
            let name = decorator_name(decorator);
            if let Some(existing) = semantics.decorators.get(&name) {
                if existing.owner != entry.canonical {
                    return Err(error(
                        decorator.name.span,
                        format!(
                            "semantic decorator `@{name}` is owned by both `{}` and `{}`",
                            existing.owner, entry.canonical
                        ),
                    ));
                }
                continue;
            }
            semantics.decorators.insert(
                name,
                TraitDecoratorDefinition {
                    owner: entry.canonical.clone(),
                    policies: decorator
                        .symbols
                        .iter()
                        .filter_map(|symbol| {
                            symbol
                                .value
                                .as_ref()
                                .map(|value| (symbol.spelling.clone(), value.clone()))
                        })
                        .collect(),
                },
            );
        }
    }

    for key in keys {
        let mut namespace = TraitSemanticNamespace::default();
        collect_trait_semantic_namespace(&key, registry, &mut HashSet::new(), &mut namespace)?;
        semantics.namespaces.insert(key, namespace);
    }
    Ok(semantics)
}

fn collect_trait_semantic_namespace(
    key: &str,
    registry: &HashMap<String, TraitEntry>,
    visited: &mut HashSet<String>,
    namespace: &mut TraitSemanticNamespace,
) -> Result<(), SemanticError> {
    if !visited.insert(key.to_owned()) {
        return Ok(());
    }
    let entry = registry
        .get(key)
        .expect("semantic trait keys come from the trait registry");
    namespace.traits.push(entry.canonical.clone());
    for operator in &entry.declaration.operators {
        push_trait_provider(&mut namespace.operators, &operator.symbol, &entry.canonical);
    }
    for method in &entry.declaration.methods {
        push_trait_provider(
            &mut namespace.operations,
            &method.name.name,
            &entry.canonical,
        );
    }
    let mut with_behavior = None;
    let mut without_behavior = None;
    for behavior in &entry.declaration.scoped_behaviors {
        let slot = match behavior.phase {
            severian_ast::TraitScopedBehaviorPhase::With => &mut with_behavior,
            severian_ast::TraitScopedBehaviorPhase::Without => &mut without_behavior,
        };
        if slot.replace(behavior.clone()).is_some() {
            return Err(error(
                behavior.span,
                format!(
                    "E000211: trait `{}` declares the same scoped behavior phase more than once",
                    entry.canonical
                ),
            ));
        }
    }
    match (with_behavior, without_behavior) {
        (Some(with_behavior), Some(without_behavior)) => {
            let with_type = with_behavior.params[0]
                .ty
                .as_ref()
                .map(declaration_type_key);
            let without_type = without_behavior.params[0]
                .ty
                .as_ref()
                .map(declaration_type_key);
            if with_type != without_type {
                return Err(error(
                    without_behavior.params[0].span,
                    format!(
                        "E000211: trait `{}` must use the same `context` type for `with` and `without`",
                        entry.canonical
                    ),
                ));
            }
            namespace
                .scoped_behaviors
                .push(TraitScopedBehaviorProvider {
                    trait_name: entry.canonical.clone(),
                });
        }
        (Some(behavior), None) | (None, Some(behavior)) => {
            return Err(error(
                behavior.span,
                format!(
                    "E000211: trait `{}` must declare both `with(context)` and `without(context)`",
                    entry.canonical
                ),
            ));
        }
        (None, None) => {}
    }
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
        collect_trait_semantic_namespace(&composed_key, registry, visited, namespace)?;
    }
    Ok(())
}

fn push_trait_provider(
    members: &mut BTreeMap<String, Vec<TraitMemberProvider>>,
    member: &str,
    trait_name: &str,
) {
    let provider = TraitMemberProvider {
        trait_name: trait_name.to_owned(),
        qualified_member: format!("{trait_name}::{member}"),
    };
    let providers = members.entry(member.to_owned()).or_default();
    if !providers.contains(&provider) {
        providers.push(provider);
    }
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
    let mut property_types = expanded
        .properties
        .iter()
        .map(|property| {
            (
                property.name.name.clone(),
                declaration_type_key(&property.ty),
            )
        })
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
        for property in &inherited.properties {
            let property = substitute_trait_property(property, &substitutions);
            let property_type = declaration_type_key(&property.ty);
            match property_types.get(&property.name.name) {
                Some(existing) if existing == &property_type => continue,
                Some(existing) => {
                    return Err(error(
                        composed.span(),
                        format!(
                            "trait `{}` composes conflicting property requirements for `{}`: `{existing}` and `{property_type}`",
                            entry.declaration.name.name, property.name.name
                        ),
                    ))
                }
                None => {
                    property_types.insert(property.name.name.clone(), property_type);
                    expanded.properties.push(property);
                }
            }
        }
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

fn substitute_trait_property(
    property: &severian_ast::TraitProperty,
    substitutions: &HashMap<String, Type>,
) -> severian_ast::TraitProperty {
    severian_ast::TraitProperty {
        ty: substitute_declared_type(&property.ty, substitutions),
        ..property.clone()
    }
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
) -> Result<BTreeMap<String, TraitRegistryDefinition>, SemanticError> {
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
            validate_trait_properties(class, &raw, declaration, &substitutions)?;
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
    build_trait_implementation_registries(module, interfaces, aliases)
}

fn validate_trait_properties(
    class: &severian_ast::ClassDecl,
    trait_name: &str,
    declaration: &severian_ast::TraitDecl,
    substitutions: &HashMap<String, Type>,
) -> Result<(), SemanticError> {
    for required in &declaration.properties {
        let expected = substitute_declared_type(&required.ty, substitutions);
        let field = class
            .fields
            .iter()
            .find(|field| field.name.name == required.name.name);
        let value = field
            .and_then(|field| field.default.as_ref())
            .or(required.default.as_ref())
            .ok_or_else(|| {
                error(
                    class.name.span,
                    format!(
                        "E000212: class `{}` does not contribute property `{}` required by trait `{trait_name}`",
                        class.name.name, required.name.name
                    ),
                )
            })?;
        if let Some(actual) = field.and_then(|field| field.ty.as_ref()) {
            if !optional_declaration_types_match(Some(actual), Some(&expected)) {
                return Err(error(
                    actual.span(),
                    format!(
                        "E000212: property `{}.{}` has type `{}`, expected `{}` from trait `{trait_name}`",
                        class.name.name,
                        required.name.name,
                        declaration_type_key(actual),
                        declaration_type_key(&expected)
                    ),
                ));
            }
        } else if !trait_property_expression_matches_type(value, &expected) {
            return Err(error(
                value.span(),
                format!(
                    "E000212: property `{}.{}` does not match `{}` required by trait `{trait_name}`",
                    class.name.name,
                    required.name.name,
                    declaration_type_key(&expected)
                ),
            ));
        }
        trait_property_value(value)?;
    }
    Ok(())
}

fn build_trait_implementation_registries(
    module: &Module,
    interfaces: &[PackageInterface],
    root_aliases: &HashMap<String, String>,
) -> Result<BTreeMap<String, TraitRegistryDefinition>, SemanticError> {
    let mut traits = HashMap::<String, &severian_ast::TraitDecl>::new();
    let mut trait_aliases = HashMap::<String, String>::new();
    for item in &module.items {
        let Item::Trait(declaration) = item else {
            continue;
        };
        let canonical = declaration.name.name.clone();
        traits.insert(canonical.clone(), declaration);
        trait_aliases.insert(canonical.clone(), canonical);
    }
    for interface in interfaces {
        for item in &interface.module.items {
            let Item::Trait(declaration) = item else {
                continue;
            };
            let canonical = format!("{}.{}", interface.name, declaration.name.name);
            traits.insert(canonical.clone(), declaration);
            trait_aliases.insert(canonical.clone(), canonical.clone());
            if let Some(package) = &interface.export_package {
                trait_aliases.insert(
                    format!("{package}.{}", declaration.name.name),
                    canonical.clone(),
                );
            }
        }
    }

    let mut registries = BTreeMap::new();
    let mut canonical_traits = traits.keys().cloned().collect::<Vec<_>>();
    canonical_traits.sort();
    for canonical in canonical_traits {
        let declaration = traits[&canonical];
        if declaration.properties.is_empty() {
            continue;
        }
        let properties = declaration
            .properties
            .iter()
            .map(|property| {
                Ok(TraitPropertyDefinition {
                    name: property.name.name.clone(),
                    ty: declaration_type_key(&property.ty),
                    default: property
                        .default
                        .as_ref()
                        .map(trait_property_value)
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, SemanticError>>()?;
        registries.insert(
            canonical.clone(),
            TraitRegistryDefinition {
                name: canonical,
                properties,
                implementations: Vec::new(),
            },
        );
    }

    collect_registry_implementations(
        &module.items,
        None,
        root_aliases,
        &traits,
        &trait_aliases,
        &mut registries,
    )?;
    for interface in interfaces {
        let aliases = collect_imports(&interface.module);
        collect_registry_implementations(
            &interface.module.items,
            Some(&interface.name),
            &aliases,
            &traits,
            &trait_aliases,
            &mut registries,
        )?;
    }

    for registry in registries.values_mut() {
        registry
            .implementations
            .sort_by(|left, right| left.name.cmp(&right.name));
    }
    Ok(registries)
}

fn collect_registry_implementations(
    items: &[Item],
    namespace: Option<&str>,
    aliases: &HashMap<String, String>,
    traits: &HashMap<String, &severian_ast::TraitDecl>,
    trait_aliases: &HashMap<String, String>,
    registries: &mut BTreeMap<String, TraitRegistryDefinition>,
) -> Result<(), SemanticError> {
    for item in items {
        let Item::Class(class) = item else { continue };
        for implemented in &class.traits {
            let Some(raw) = declaration_type_name(implemented) else {
                continue;
            };
            if raw.rsplit('.').next() == Some("From") {
                continue;
            }
            let Some(canonical) =
                resolve_registry_trait(&raw, namespace, aliases, traits, trait_aliases)
            else {
                continue;
            };
            let declaration = traits[&canonical];
            let substitutions = trait_substitutions(declaration, implemented, class)?;
            validate_trait_properties(class, &raw, declaration, &substitutions)?;
            let provider = match namespace {
                Some(namespace) => format!("{namespace}.{}", class.name.name),
                None => class.name.name.clone(),
            };
            let mut implemented_traits = Vec::new();
            collect_registry_trait_closure(
                &canonical,
                aliases,
                traits,
                trait_aliases,
                &mut HashSet::new(),
                &mut implemented_traits,
            );
            for implemented_trait in implemented_traits {
                let Some(registry) = registries.get_mut(&implemented_trait) else {
                    continue;
                };
                let required_properties = &traits[&implemented_trait].properties;
                let mut properties = BTreeMap::new();
                for required in required_properties {
                    let field = class
                        .fields
                        .iter()
                        .find(|field| field.name.name == required.name.name);
                    let inherited = declaration
                        .properties
                        .iter()
                        .find(|property| property.name.name == required.name.name);
                    let value = field
                        .and_then(|field| field.default.as_ref())
                        .or_else(|| inherited.and_then(|property| property.default.as_ref()))
                        .or(required.default.as_ref())
                        .expect("trait property validation requires a provider value");
                    properties.insert(required.name.name.clone(), trait_property_value(value)?);
                }
                add_trait_registry_provider(registry, &provider, properties, implemented.span())?;
            }
        }
    }
    Ok(())
}

fn collect_registry_trait_closure(
    canonical: &str,
    aliases: &HashMap<String, String>,
    traits: &HashMap<String, &severian_ast::TraitDecl>,
    trait_aliases: &HashMap<String, String>,
    visited: &mut HashSet<String>,
    output: &mut Vec<String>,
) {
    if !visited.insert(canonical.to_owned()) {
        return;
    }
    output.push(canonical.to_owned());
    let namespace = canonical.rsplit_once('.').map(|(namespace, _)| namespace);
    for composed in &traits[canonical].composed_traits {
        let Some(raw) = declaration_type_name(composed) else {
            continue;
        };
        let Some(composed) =
            resolve_registry_trait(&raw, namespace, aliases, traits, trait_aliases)
        else {
            continue;
        };
        collect_registry_trait_closure(&composed, aliases, traits, trait_aliases, visited, output);
    }
}

fn add_trait_registry_provider(
    registry: &mut TraitRegistryDefinition,
    provider: &str,
    properties: BTreeMap<String, TraitPropertyValue>,
    span: Span,
) -> Result<(), SemanticError> {
    if registry
        .implementations
        .iter()
        .any(|implementation| implementation.name == provider)
    {
        return Ok(());
    }
    for existing in &registry.implementations {
        for (property, value) in &properties {
            let Some(existing_value) = existing.properties.get(property) else {
                continue;
            };
            if trait_property_values_overlap(existing_value, value) {
                return Err(error(
                    span,
                    format!(
                        "E000212: trait registry `{}` has an ambiguous `{property}` contribution shared by `{}` and `{provider}`",
                        registry.name, existing.name
                    ),
                ));
            }
        }
    }
    registry
        .implementations
        .push(TraitImplementationDefinition {
            name: provider.to_owned(),
            properties,
        });
    Ok(())
}

fn resolve_registry_trait(
    raw: &str,
    namespace: Option<&str>,
    aliases: &HashMap<String, String>,
    traits: &HashMap<String, &severian_ast::TraitDecl>,
    trait_aliases: &HashMap<String, String>,
) -> Option<String> {
    let canonical = canonical_declared_type_name(raw, aliases);
    for candidate in [canonical.as_str(), raw] {
        if traits.contains_key(candidate) {
            return Some(candidate.to_owned());
        }
        if let Some(canonical) = trait_aliases.get(candidate) {
            return Some(canonical.clone());
        }
    }
    if !raw.contains('.') {
        if let Some(namespace) = namespace {
            let candidate = format!("{namespace}.{raw}");
            if traits.contains_key(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn trait_property_values_overlap(left: &TraitPropertyValue, right: &TraitPropertyValue) -> bool {
    match (left, right) {
        (TraitPropertyValue::Set(left), TraitPropertyValue::Set(right)) => {
            left.iter().any(|value| right.contains(value))
        }
        _ => left == right,
    }
}

fn trait_property_expression_matches_type(value: &Expr, expected: &Type) -> bool {
    let expected = declaration_type_key(expected);
    inferred_trait_property_type(value).is_some_and(|actual| actual == expected)
}

fn inferred_trait_property_type(value: &Expr) -> Option<String> {
    match value {
        Expr::Literal(Literal::Integer { .. }) => Some("int".into()),
        Expr::Literal(Literal::Float { .. }) => Some("float".into()),
        Expr::Literal(Literal::Boolean { .. }) => Some("bool".into()),
        Expr::Literal(Literal::String { .. }) => Some("string".into()),
        Expr::Literal(Literal::Null { .. }) => None,
        Expr::List(collection) => homogeneous_collection_type("list", &collection.elements),
        Expr::Set(collection) => homogeneous_collection_type("set", &collection.elements),
        Expr::Tuple(collection) => Some(format!(
            "tuple[{}]",
            collection
                .elements
                .iter()
                .map(inferred_trait_property_type)
                .collect::<Option<Vec<_>>>()?
                .join(", ")
        )),
        Expr::Map(map) => {
            let first = map.entries.first()?;
            let key = inferred_trait_property_type(&first.key)?;
            let value = inferred_trait_property_type(&first.value)?;
            map.entries
                .iter()
                .all(|entry| {
                    inferred_trait_property_type(&entry.key).as_deref() == Some(key.as_str())
                        && inferred_trait_property_type(&entry.value).as_deref()
                            == Some(value.as_str())
                })
                .then(|| format!("map[{key}, {value}]"))
        }
        Expr::Call(call) => expression_path(&call.callee)
            .and_then(|path| path.rsplit('.').next().map(str::to_owned)),
        Expr::Member(member) => expression_path(&member.object)
            .and_then(|path| path.rsplit('.').next().map(str::to_owned)),
        Expr::Identifier(identifier) => Some(identifier.name.clone()),
        _ => None,
    }
}

fn homogeneous_collection_type(kind: &str, elements: &[Expr]) -> Option<String> {
    let first = inferred_trait_property_type(elements.first()?)?;
    elements
        .iter()
        .all(|element| inferred_trait_property_type(element).as_deref() == Some(first.as_str()))
        .then(|| format!("{kind}[{first}]"))
}

fn trait_property_value(value: &Expr) -> Result<TraitPropertyValue, SemanticError> {
    let constant = match value {
        Expr::Literal(Literal::Integer { value, .. }) => TraitPropertyValue::Integer(*value),
        Expr::Literal(Literal::Float { value, .. }) => TraitPropertyValue::Float(value.to_bits()),
        Expr::Literal(Literal::Boolean { value, .. }) => TraitPropertyValue::Boolean(*value),
        Expr::Literal(Literal::String { value, .. }) => TraitPropertyValue::String(value.clone()),
        Expr::Identifier(identifier) => TraitPropertyValue::Symbol(identifier.name.clone()),
        Expr::Member(_) => TraitPropertyValue::Symbol(
            expression_path(value).expect("member expressions always have a path here"),
        ),
        Expr::Call(call) => {
            let name = expression_path(&call.callee).ok_or_else(|| {
                error(
                    call.callee.span(),
                    "E000212: trait property constructors must use a named type",
                )
            })?;
            if call.args.iter().any(|argument| argument.name.is_some()) {
                return Err(error(
                    call.span,
                    "E000212: trait property constructors do not accept named arguments",
                ));
            }
            TraitPropertyValue::Constructor {
                name,
                arguments: call
                    .args
                    .iter()
                    .map(|argument| trait_property_value(&argument.value))
                    .collect::<Result<Vec<_>, _>>()?,
            }
        }
        Expr::List(collection) => TraitPropertyValue::List(
            collection
                .elements
                .iter()
                .map(trait_property_value)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Set(collection) => {
            let mut values = collection
                .elements
                .iter()
                .map(trait_property_value)
                .collect::<Result<Vec<_>, _>>()?;
            values.sort();
            values.dedup();
            TraitPropertyValue::Set(values)
        }
        Expr::Tuple(collection) => TraitPropertyValue::Tuple(
            collection
                .elements
                .iter()
                .map(trait_property_value)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Map(map) => {
            let mut entries = map
                .entries
                .iter()
                .map(|entry| {
                    Ok((
                        trait_property_value(&entry.key)?,
                        trait_property_value(&entry.value)?,
                    ))
                })
                .collect::<Result<Vec<_>, SemanticError>>()?;
            entries.sort();
            TraitPropertyValue::Map(entries)
        }
        _ => {
            return Err(error(
                value.span(),
                "E000212: trait registry properties must be compile-time constants",
            ))
        }
    };
    Ok(constant)
}

fn expression_path(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Identifier(identifier) => Some(identifier.name.clone()),
        Expr::Member(member) => Some(format!(
            "{}.{}",
            expression_path(&member.object)?,
            member.member.name
        )),
        _ => None,
    }
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
        ("@", [left, right]) => left == right && returns == *left && left.starts_with("Tensor["),
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
