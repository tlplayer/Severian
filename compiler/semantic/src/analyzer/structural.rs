use super::*;

pub(super) fn register_concrete_trait_aliases(
    aliases: &mut HashMap<String, String>,
    module: &Module,
    interfaces: &[PackageInterface],
) -> Result<(), SemanticError> {
    let traits = module
        .items
        .iter()
        .chain(
            interfaces
                .iter()
                .flat_map(|interface| interface.module.items.iter()),
        )
        .filter_map(|item| match item {
            Item::Trait(declaration) => Some((declaration.name.name.clone(), declaration.clone())),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut declared_types = Vec::new();
    for item in &module.items {
        collect_item_types(item, &mut declared_types);
    }
    for ty in declared_types {
        register_concrete_trait_type(aliases, &traits, &ty)?;
    }
    Ok(())
}

fn register_concrete_trait_type(
    aliases: &mut HashMap<String, String>,
    traits: &HashMap<String, severian_ast::TraitDecl>,
    ty: &Type,
) -> Result<(), SemanticError> {
    let Type::Named(path) = ty else {
        return Ok(());
    };
    for argument in &path.args {
        if let Some(argument) = argument.as_type() {
            register_concrete_trait_type(aliases, traits, argument)?;
        }
    }
    let Some(name) = path.segments.last().map(|segment| segment.name.as_str()) else {
        return Ok(());
    };
    let Some(declaration) = traits.get(name) else {
        return Ok(());
    };
    if declaration.generic_params.is_empty() || path.args.is_empty() {
        return Ok(());
    }
    if path.args.len() != declaration.generic_params.len() {
        return Err(error(
            path.span,
            format!(
                "trait `{name}` expects {} type argument(s), received {}",
                declaration.generic_params.len(),
                path.args.len()
            ),
        ));
    }
    let arguments = path
        .args
        .iter()
        .map(|argument| argument.as_type().cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| error(path.span, format!("trait `{name}` requires type arguments")))?;
    let substitutions = declaration
        .generic_params
        .iter()
        .zip(arguments)
        .map(|(parameter, argument)| (parameter.name.name.clone(), argument))
        .collect::<HashMap<_, _>>();
    let concrete = declaration_type_key(ty);
    aliases.insert(format!("__trait.{concrete}"), String::new());
    aliases.insert(
        format!("__class_methods.{concrete}"),
        declaration
            .methods
            .iter()
            .map(|method| method.name.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    for method in &declaration.methods {
        let params = method
            .params
            .iter()
            .map(|parameter| severian_ast::Parameter {
                ty: parameter
                    .ty
                    .as_ref()
                    .map(|ty| substitute_declared_type(ty, &substitutions)),
                ..parameter.clone()
            })
            .collect::<Vec<_>>();
        let returns = method
            .return_type
            .as_ref()
            .map(|ty| substitute_declared_type(ty, &substitutions));
        aliases.insert(
            format!("__class_method_signature.{concrete}.{}", method.name.name),
            callable_signature(&params, returns.as_ref()),
        );
        register_method_return_alias(aliases, &concrete, &method.name.name, returns.as_ref())?;
    }
    Ok(())
}

fn collect_item_types(item: &Item, output: &mut Vec<Type>) {
    match item {
        Item::Function(function) => collect_function_types(function, output),
        Item::Class(class) => {
            output.extend(class.traits.iter().cloned());
            output.extend(class.fields.iter().filter_map(|field| field.ty.clone()));
            for method in &class.methods {
                collect_function_types(method, output);
            }
        }
        Item::Trait(declaration) => {
            for method in &declaration.methods {
                output.extend(
                    method
                        .params
                        .iter()
                        .filter_map(|parameter| parameter.ty.clone()),
                );
                output.extend(method.return_type.clone());
            }
        }
        Item::Enum(enumeration) => {
            for variant in &enumeration.variants {
                output.extend(variant.fields.iter().filter_map(|field| field.ty.clone()));
            }
        }
        Item::Statement(_) | Item::Import(_) => {}
    }
}

fn collect_function_types(function: &severian_ast::FunctionDecl, output: &mut Vec<Type>) {
    output.extend(
        function
            .params
            .iter()
            .filter_map(|parameter| parameter.ty.clone()),
    );
    output.extend(function.return_type.clone());
}

pub(super) fn substitute_declared_type(ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
    if let Type::Named(path) = ty {
        if path.args.is_empty() && path.segments.len() == 1 {
            if let Some(replacement) = substitutions.get(&path.segments[0].name) {
                return replacement.clone();
            }
        }
    }
    let mut substituted = ty.clone();
    match &mut substituted {
        Type::Named(path) => {
            for argument in &mut path.args {
                if let TypeArg::Type { ty, .. } = argument {
                    **ty = substitute_declared_type(ty, substitutions);
                }
            }
        }
        Type::List { element, .. }
        | Type::Set { element, .. }
        | Type::Option { some: element, .. }
        | Type::Future {
            output: element, ..
        }
        | Type::Reference { inner: element, .. } => {
            **element = substitute_declared_type(element, substitutions)
        }
        Type::Tuple { elements, .. }
        | Type::Union {
            alternatives: elements,
            ..
        } => {
            for element in elements {
                *element = substitute_declared_type(element, substitutions);
            }
        }
        Type::Map { key, value, .. }
        | Type::Result {
            ok: key,
            err: value,
            ..
        } => {
            **key = substitute_declared_type(key, substitutions);
            **value = substitute_declared_type(value, substitutions);
        }
        Type::Function {
            params, returns, ..
        } => {
            for parameter in params {
                *parameter = substitute_declared_type(parameter, substitutions);
            }
            **returns = substitute_declared_type(returns, substitutions);
        }
    }
    substituted
}
