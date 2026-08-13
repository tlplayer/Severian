use super::*;

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
                            optional_declaration_types_match(
                                actual.ty.as_ref(),
                                expected.ty.as_ref(),
                            )
                        });
                let return_matches = optional_declaration_types_match(
                    method.return_type.as_ref(),
                    required.return_type.as_ref(),
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
        }
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
