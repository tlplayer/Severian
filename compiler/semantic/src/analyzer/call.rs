use super::*;

pub(super) fn lower_call(
    call: &severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    if let Some((class, assignments)) = completed_builder(call, scope, aliases)? {
        let fields =
            lower_construction_fields(call.span, &class, assignments, scope, signatures, aliases)?;
        return Ok((
            Expression::ConstructFields {
                type_id: TypeDefinitionId::from_name(&class),
                class,
                fields,
                validate: true,
            },
            ValueType::Any,
        ));
    }

    if let Expr::Member(member) = call.callee.as_ref() {
        if member.member.name == "from" {
            if let Some(target_class) = static_class_name(&member.object, aliases) {
                if call.args.len() != 1 || call.args[0].name.is_some() {
                    return Err(error(call.span, "`from` expects exactly one source value"));
                }
                let source_class = expression_class(&call.args[0].value, scope, aliases)
                    .ok_or_else(|| {
                        error(
                            call.args[0].span,
                            "structural conversion requires a statically known source class",
                        )
                    })?;
                if class_methods(&target_class, aliases).contains(&"from") {
                    let source =
                        lower_expression(&call.args[0].value, scope, signatures, aliases)?.0;
                    return Ok((
                        Expression::MethodCall {
                            object: Box::new(Expression::ConstructFields {
                                type_id: TypeDefinitionId::from_name(&target_class),
                                class: target_class.clone(),
                                fields: Vec::new(),
                                validate: false,
                            }),
                            method: "from".into(),
                            args: vec![source],
                        },
                        ValueType::Any,
                    ));
                }
                if class_implements_trait(&source_class, "Document", aliases) {
                    let source =
                        lower_expression(&call.args[0].value, scope, signatures, aliases)?.0;
                    return Ok((
                        Expression::ObjectUpdate {
                            object: Box::new(Expression::MethodCall {
                                object: Box::new(source),
                                method: "value".into(),
                                args: Vec::new(),
                            }),
                            type_id: TypeDefinitionId::from_name(&target_class),
                            class: target_class,
                            fields: Vec::new(),
                            json_document: true,
                        },
                        ValueType::Any,
                    ));
                }
                validate_structural_conversion(call.span, &source_class, &target_class, aliases)?;
                let source = lower_expression(&call.args[0].value, scope, signatures, aliases)?.0;
                return Ok((
                    Expression::ObjectUpdate {
                        object: Box::new(source),
                        type_id: TypeDefinitionId::from_name(&target_class),
                        class: target_class,
                        fields: Vec::new(),
                        json_document: false,
                    },
                    ValueType::Any,
                ));
            }
        }

        if member.member.name == "into" {
            if call.args.len() != 1 || call.args[0].name.is_some() {
                return Err(error(call.span, "`into` expects exactly one target type"));
            }
            let target_class = static_class_name(&call.args[0].value, aliases)
                .ok_or_else(|| error(call.args[0].span, "`into` requires a class name"))?;
            let source_class =
                expression_class(&member.object, scope, aliases).ok_or_else(|| {
                    error(
                        member.object.span(),
                        "structural conversion requires a statically known source class",
                    )
                })?;
            if class_methods(&target_class, aliases).contains(&"from") {
                let source = lower_expression(&member.object, scope, signatures, aliases)?.0;
                return Ok((
                    Expression::MethodCall {
                        object: Box::new(Expression::ConstructFields {
                            type_id: TypeDefinitionId::from_name(&target_class),
                            class: target_class.clone(),
                            fields: Vec::new(),
                            validate: false,
                        }),
                        method: "from".into(),
                        args: vec![source],
                    },
                    ValueType::Any,
                ));
            }
            if class_implements_trait(&target_class, "Document", aliases) {
                let source = lower_expression(&member.object, scope, signatures, aliases)?.0;
                let document = Expression::ObjectDocument {
                    object: Box::new(source),
                    fields: class_fields(&source_class, aliases)
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                };
                let declared_fields = class_fields(&target_class, aliases);
                let payload = declared_fields
                    .iter()
                    .copied()
                    .find(|field| *field != "source_path")
                    .ok_or_else(|| {
                        error(
                            call.span,
                            format!("document adapter `{target_class}` needs a payload field"),
                        )
                    })?;
                let mut fields = vec![(payload.to_owned(), document)];
                if declared_fields.contains(&"source_path") {
                    fields.push(("source_path".into(), Expression::String(String::new())));
                }
                return Ok((
                    Expression::ConstructFields {
                        type_id: TypeDefinitionId::from_name(&target_class),
                        class: target_class,
                        fields,
                        validate: true,
                    },
                    ValueType::Any,
                ));
            }
            validate_structural_conversion(call.span, &source_class, &target_class, aliases)?;
            let source = lower_expression(&member.object, scope, signatures, aliases)?.0;
            return Ok((
                Expression::ObjectUpdate {
                    object: Box::new(source),
                    type_id: TypeDefinitionId::from_name(&target_class),
                    class: target_class,
                    fields: Vec::new(),
                    json_document: false,
                },
                ValueType::Any,
            ));
        }

        if member.member.name == "with" {
            if let Some(class) = expression_class(&member.object, scope, aliases) {
                if call.args.iter().any(|argument| argument.name.is_none()) {
                    return Err(error(call.span, "`with` accepts only named field updates"));
                }
                let assignments = call
                    .args
                    .iter()
                    .map(|argument| {
                        (
                            argument.name.as_ref().unwrap().name.clone(),
                            &argument.value,
                        )
                    })
                    .collect();
                let fields = lower_update_fields(
                    call.span,
                    &class,
                    assignments,
                    scope,
                    signatures,
                    aliases,
                )?;
                let source = lower_expression(&member.object, scope, signatures, aliases)?.0;
                return Ok((
                    Expression::ObjectUpdate {
                        object: Box::new(source),
                        type_id: TypeDefinitionId::from_name(&class),
                        class,
                        fields,
                        json_document: false,
                    },
                    ValueType::Any,
                ));
            }
        }
    }

    if let Expr::Index(index) = call.callee.as_ref() {
        if let Expr::Identifier(callee) = index.object.as_ref() {
            let imported = aliases
                .get(&callee.name)
                .map(String::as_str)
                .unwrap_or(&callee.name);
            let canonical = resolve_linked_function(imported, signatures);
            if let Some(signature) = signatures.get(canonical) {
                return lower_declared_call(call, canonical, signature, scope, signatures, aliases);
            }
        }
        if let Expr::Member(member) = index.object.as_ref() {
            if let Expr::Identifier(object) = member.object.as_ref() {
                if let Some(module) = aliases.get(&object.name) {
                    let function = format!("{module}.{}", member.member.name);
                    let canonical = resolve_linked_function(&function, signatures);
                    let signature = signatures.get(canonical).ok_or_else(|| {
                        error(call.span, format!("unknown exported function `{function}`"))
                    })?;
                    return lower_declared_call(
                        call, canonical, signature, scope, signatures, aliases,
                    );
                }
            }
        }
    }
    if let Expr::Member(member) = call.callee.as_ref() {
        if let Expr::Identifier(object) = member.object.as_ref() {
            if (object.name == "int" || object.name == "float") && member.member.name == "parse" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        target: CallTarget::source(object.name.clone() + ".parse"),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if object.name == "http" && member.member.name == "get" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("http.get"),
                        args,
                    },
                    ValueType::Result,
                ));
            }
            if member.member.name == "zero" && !scope.contains_key(&object.name) {
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("Number.zero"),
                        args: Vec::new(),
                    },
                    ValueType::Any,
                ));
            }
            if let Some(module) = aliases.get(&object.name) {
                let function = format!("{module}.{}", member.member.name);
                let canonical = resolve_linked_function(&function, signatures);
                if let Some(signature) = signatures.get(canonical) {
                    return lower_declared_call(
                        call, canonical, signature, scope, signatures, aliases,
                    );
                }
                if let Some(class) = aliases.get(&format!("__module_class.{function}")) {
                    return lower_class_invocation(call, class, scope, signatures, aliases);
                }
                return Err(error(
                    call.span,
                    format!("unknown exported function or class `{function}`"),
                ));
            }
        }
        let object_class = expression_class(&member.object, scope, aliases);
        let known_class_method = object_class.as_ref().is_some_and(|class| {
            aliases
                .get(&format!("__class_methods.{class}"))
                .is_some_and(|methods| {
                    methods
                        .split(',')
                        .any(|method| method == member.member.name)
                })
        });
        let dynamic_object_access =
            matches!(member.member.name.as_str(), "get" | "set") && !known_class_method;
        let (object, object_type) = lower_expression(&member.object, scope, signatures, aliases)?;
        let lowered_args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases))
            .collect::<Result<Vec<_>, _>>()?;
        if object_type == ValueType::Any && member.member.name == "set" && dynamic_object_access {
            if let Expr::Identifier(identifier) = member.object.as_ref() {
                if !scope
                    .get(&identifier.name)
                    .is_some_and(|binding| binding.mutable || binding.field)
                {
                    return Err(error(
                        member.object.span(),
                        format!(
                            "object `{}` is not changeable; bind it with `:=` before calling `set`",
                            identifier.name
                        ),
                    ));
                }
            }
        }
        if !known_class_method {
            validate_builtin_method(
                call.span,
                object_type,
                &member.member.name,
                &lowered_args.iter().map(|(_, ty)| *ty).collect::<Vec<_>>(),
            )?;
        }
        let known_field = dynamic_object_access
            .then_some(())
            .and_then(|()| object_class.as_ref())
            .filter(|class| !aliases.contains_key(&format!("__trait.{class}")))
            .and_then(|class| {
                let Expr::Literal(Literal::String { value, .. }) =
                    call.args.first().map(|argument| &argument.value)?
                else {
                    return None;
                };
                Some((class, value))
            });
        let field_type = if let Some((class, field)) = known_field {
            let fields = aliases
                .get(&format!("__class_fields.{class}"))
                .map(|fields| fields.split(',').collect::<Vec<_>>())
                .unwrap_or_default();
            if !fields.contains(&field.as_str()) {
                return Err(error(
                    call.args[0].value.span(),
                    format!("class `{class}` has no field `{field}`"),
                ));
            }
            aliases
                .get(&format!("__class_field_type.{class}.{field}"))
                .and_then(|value| decode_field_type(value))
        } else {
            None
        };
        if member.member.name == "set" && dynamic_object_access {
            if let (Some(expected), Some((_, actual))) = (field_type, lowered_args.get(1)) {
                compatible(call.args[1].value.span(), *actual, expected)?;
            }
        }
        let args = lowered_args.into_iter().map(|(arg, _)| arg).collect();
        let declared_method_return = object_class.as_ref().and_then(|class| {
            aliases
                .get(&format!(
                    "__class_method_return.{class}.{}",
                    member.member.name
                ))
                .and_then(|value| decode_field_type(value))
        });
        let return_type = if member.member.name == "get" && dynamic_object_access {
            field_type.unwrap_or(ValueType::Any)
        } else {
            declared_method_return
                .unwrap_or_else(|| method_return_type(object_type, &member.member.name))
        };
        return Ok((
            Expression::MethodCall {
                object: Box::new(object),
                method: member.member.name.clone(),
                args,
            },
            return_type,
        ));
    }
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        let (callee, ty) = lower_expression(&call.callee, scope, signatures, aliases)?;
        if ty != ValueType::Function {
            return Err(error(call.callee.span(), "value is not callable"));
        }
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(callee),
                args,
                return_type: ValueType::Any,
            },
            ValueType::Any,
        ));
    };
    let intrinsic = matches!(
        callee.name.as_str(),
        "print"
            | "panic"
            | "float"
            | "string"
            | "range"
            | "indices"
            | "enumerate"
            | "zip"
            | "any"
            | "all"
            | "abs"
            | "min"
            | "max"
            | "divmod"
            | "len"
            | "size"
            | "bytes"
            | "bits"
            | "capacity"
    );
    let imported = if intrinsic {
        callee.name.as_str()
    } else if signatures.contains_key(&callee.name) {
        callee.name.as_str()
    } else {
        aliases
            .get(&callee.name)
            .map(String::as_str)
            .unwrap_or(&callee.name)
    };
    // Path dependencies are compiled into the same translation unit. Their
    // functions therefore have source-level names, while `from package import`
    // records a package-qualified alias. Prefer a real qualified interface when
    // one exists, then fall back to the linked source implementation.
    let canonical = resolve_linked_function(imported, signatures);
    if let Some(class) = aliases.get(&format!("__module_class.{imported}")) {
        return lower_class_invocation(call, class, scope, signatures, aliases);
    }
    if let Some(class) = canonical.strip_prefix("__class.") {
        return lower_class_invocation(call, class, scope, signatures, aliases);
    }
    let builtin = match canonical {
        "print" | "io.print" => Some(("print", ValueType::Unit)),
        "panic" => Some(("panic", ValueType::Unit)),
        "float" => Some(("float", ValueType::Float)),
        "string" => Some(("string", ValueType::String)),
        "range" => Some(("range", ValueType::List)),
        "indices" => Some(("indices", ValueType::List)),
        "enumerate" => Some(("enumerate", ValueType::List)),
        "zip" => Some(("zip", ValueType::List)),
        "any" => Some(("any", ValueType::Bool)),
        "all" => Some(("all", ValueType::Bool)),
        "abs" | "min" | "max" => Some((canonical, ValueType::Any)),
        "divmod" => Some(("divmod", ValueType::Tuple)),
        "read" if !signatures.contains_key(&callee.name) => Some(("read", ValueType::Result)),
        "len" | "size" | "bytes" | "bits" | "capacity" => Some((canonical, ValueType::Int)),
        "present" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    type_id: Some(TypeDefinitionId::from_name("Option")),
                    variant_id: VariantId::from_name("present"),
                    name: "present".into(),
                    fields,
                },
                ValueType::Option,
            ));
        }
        "failure" => {
            let fields = call
                .args
                .iter()
                .map(|arg| {
                    lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((
                Expression::Variant {
                    type_id: Some(TypeDefinitionId::from_name("Result")),
                    variant_id: VariantId::from_name("failure"),
                    name: "failure".into(),
                    fields,
                },
                ValueType::Result,
            ));
        }
        "__format" => {
            let Expr::Literal(Literal::String { value, .. }) = &call.args[0].value else {
                unreachable!()
            };
            let (args, arg_types) = lower_format_args(value, scope, call.span)?;
            return Ok((
                Expression::Format {
                    template: value.clone(),
                    args,
                    arg_types,
                },
                ValueType::String,
            ));
        }
        _ => None,
    };
    if let Some((name, returns)) = builtin {
        let lowered = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases))
            .collect::<Result<Vec<_>, _>>()?;
        let types = lowered.iter().map(|(_, ty)| *ty).collect::<Vec<_>>();
        let valid_arity = match name {
            "range" => (1..=3).contains(&lowered.len()),
            "zip" => lowered.len() == 2,
            "min" | "max" | "divmod" => lowered.len() == 2,
            "print" | "panic" => !lowered.is_empty(),
            _ => lowered.len() == 1,
        };
        if !valid_arity {
            return Err(error(
                call.span,
                format!("builtin `{name}` received an invalid number of arguments"),
            ));
        }
        if name == "range"
            && types
                .iter()
                .any(|ty| !matches!(ty, ValueType::Int | ValueType::Any))
        {
            return Err(error(call.span, "`range` arguments must be integers"));
        }
        if matches!(name, "enumerate" | "any" | "all")
            && !matches!(
                types[0],
                ValueType::List | ValueType::Tuple | ValueType::Set
            )
        {
            return Err(error(call.span, format!("`{name}` expects an iterable")));
        }
        if name == "zip"
            && types
                .iter()
                .any(|ty| !matches!(ty, ValueType::List | ValueType::Tuple | ValueType::Set))
        {
            return Err(error(call.span, "`zip` expects two iterables"));
        }
        let args = lowered.into_iter().map(|(arg, _)| arg).collect();
        return Ok((
            Expression::Call {
                target: CallTarget::source(name),
                args,
            },
            returns,
        ));
    }
    if let Some(signature) = signatures.get(canonical) {
        return lower_declared_call(call, canonical, signature, scope, signatures, aliases);
    }
    if let Some(binding) = scope.get(&callee.name) {
        if let Some(class) = &binding.class {
            let has_forward = aliases
                .get(&format!("__class_methods.{class}"))
                .is_some_and(|methods| methods.split(',').any(|method| method == "forward"));
            if has_forward {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases)
                            .map(|(argument, _)| argument)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let return_type = aliases
                    .get(&format!("__class_method_return.{class}.forward"))
                    .and_then(|value| decode_field_type(value))
                    .unwrap_or(ValueType::Any);
                return Ok((
                    Expression::MethodCall {
                        object: Box::new(Expression::Variable(binding.reference.clone())),
                        method: "forward".into(),
                        args,
                    },
                    return_type,
                ));
            }
            return Err(error(
                call.callee.span(),
                format!("class `{class}` is not callable; define `forward` to make it callable"),
            ));
        }
    }
    if scope
        .get(&callee.name)
        .is_some_and(|binding| binding.ty == ValueType::Function)
    {
        let return_type = scope
            .get(&callee.name)
            .and_then(|binding| binding.function_return)
            .unwrap_or(ValueType::Any);
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::CallValue {
                callee: Box::new(Expression::Variable(scope[&callee.name].reference.clone())),
                args,
                return_type,
            },
            return_type,
        ));
    }
    if callee
        .name
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_uppercase)
    {
        let fields = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Variant {
                type_id: None,
                variant_id: VariantId::from_name(&callee.name),
                name: callee.name.clone(),
                fields,
            },
            ValueType::Any,
        ));
    }
    Err(error(
        callee.span,
        format!("unknown function `{}`", callee.name),
    ))
}

pub(super) fn static_class_name(
    expression: &Expr,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    match expression {
        Expr::Identifier(identifier) => aliases
            .get(&identifier.name)
            .and_then(|value| value.strip_prefix("__class."))
            .map(str::to_owned),
        Expr::Member(member) => {
            let Expr::Identifier(module) = member.object.as_ref() else {
                return None;
            };
            let direct = format!("{}.{}", module.name, member.member.name);
            aliases
                .get(&format!("__module_class.{direct}"))
                .cloned()
                .or_else(|| {
                    let module = aliases
                        .get(&module.name)
                        .map(String::as_str)
                        .unwrap_or(&module.name);
                    aliases
                        .get(&format!("__module_class.{module}.{}", member.member.name))
                        .cloned()
                })
        }
        _ => None,
    }
}

pub(super) fn generated_object_call_class(
    call: &severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    let Expr::Member(member) = call.callee.as_ref() else {
        return None;
    };
    match member.member.name.as_str() {
        "build" => builder_plan(&member.object, scope, aliases)
            .ok()
            .flatten()
            .map(|(class, _)| class),
        "from" => static_class_name(&member.object, aliases),
        "with" => expression_class(&member.object, scope, aliases),
        "into" => call
            .args
            .first()
            .and_then(|argument| static_class_name(&argument.value, aliases)),
        _ => None,
    }
}

fn class_fields<'a>(class: &str, aliases: &'a HashMap<String, String>) -> Vec<&'a str> {
    aliases
        .get(&format!("__class_fields.{class}"))
        .map(|fields| {
            fields
                .split(',')
                .filter(|field| !field.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn class_methods<'a>(class: &str, aliases: &'a HashMap<String, String>) -> Vec<&'a str> {
    aliases
        .get(&format!("__class_methods.{class}"))
        .map(|methods| {
            methods
                .split(',')
                .filter(|method| !method.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn class_implements_trait(class: &str, expected: &str, aliases: &HashMap<String, String>) -> bool {
    aliases
        .get(&format!("__class_traits.{class}"))
        .is_some_and(|traits| {
            traits.split(',').any(|implemented| {
                implemented
                    .rsplit('.')
                    .next()
                    .is_some_and(|name| name == expected)
            })
        })
}

fn class_default_fields<'a>(class: &str, aliases: &'a HashMap<String, String>) -> Vec<&'a str> {
    aliases
        .get(&format!("__class_default_fields.{class}"))
        .map(|fields| {
            fields
                .split(',')
                .filter(|field| !field.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn lower_class_invocation(
    call: &severian_ast::CallExpr,
    class: &str,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    let arity = call.args.len().to_string();
    let constructor_matches = aliases
        .get(&format!("__class_constructor_arities.{class}"))
        .is_some_and(|arities| arities.split(',').any(|candidate| candidate == arity));
    if constructor_matches && call.args.iter().all(|argument| argument.name.is_none()) {
        let args = call
            .args
            .iter()
            .map(|argument| {
                lower_expression(&argument.value, scope, signatures, aliases)
                    .map(|(argument, _)| argument)
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                type_id: TypeDefinitionId::from_name(class),
                class: class.to_owned(),
                args,
            },
            ValueType::Any,
        ));
    }

    if let [argument] = call.args.as_slice() {
        if argument.name.is_none() {
            if let Expr::Map(map) = &argument.value {
                let mut assignments = Vec::new();
                for entry in &map.entries {
                    let field = match &entry.key {
                        Expr::Identifier(identifier) => identifier.name.clone(),
                        Expr::Literal(Literal::String { value, .. }) => value.clone(),
                        _ => {
                            return Err(error(
                                entry.key.span(),
                                "class object-literal keys must be field names",
                            ))
                        }
                    };
                    assignments.push((field, &entry.value));
                }
                let fields = lower_construction_fields(
                    call.span,
                    class,
                    assignments,
                    scope,
                    signatures,
                    aliases,
                )?;
                return Ok((
                    Expression::ConstructFields {
                        type_id: TypeDefinitionId::from_name(class),
                        class: class.to_owned(),
                        fields,
                        validate: true,
                    },
                    ValueType::Any,
                ));
            }
        }
    }

    let declared_fields = class_fields(class, aliases);
    let mut named_started = false;
    let mut assignments = Vec::new();
    for (index, argument) in call.args.iter().enumerate() {
        let field = if let Some(name) = &argument.name {
            named_started = true;
            name.name.clone()
        } else {
            if named_started {
                return Err(error(
                    argument.span,
                    "positional class fields may not follow named fields",
                ));
            }
            declared_fields
                .get(index)
                .ok_or_else(|| {
                    error(
                        argument.span,
                        format!("class `{class}` received too many positional fields"),
                    )
                })?
                .to_string()
        };
        assignments.push((field, &argument.value));
    }
    let fields =
        lower_construction_fields(call.span, class, assignments, scope, signatures, aliases)?;
    Ok((
        Expression::ConstructFields {
            type_id: TypeDefinitionId::from_name(class),
            class: class.to_owned(),
            fields,
            validate: true,
        },
        ValueType::Any,
    ))
}

fn completed_builder<'a>(
    call: &'a severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Result<Option<(String, Vec<(String, &'a Expr)>)>, SemanticError> {
    let Expr::Member(member) = call.callee.as_ref() else {
        return Ok(None);
    };
    if member.member.name != "build" {
        return Ok(None);
    }
    if !call.args.is_empty() {
        return Err(error(
            call.span,
            "builder `build` does not accept arguments",
        ));
    }
    let Some(plan) = builder_plan(&member.object, scope, aliases)? else {
        return Ok(None);
    };
    Ok(Some(plan))
}

fn builder_plan<'a>(
    expression: &'a Expr,
    scope: &HashMap<String, Binding>,
    aliases: &HashMap<String, String>,
) -> Result<Option<(String, Vec<(String, &'a Expr)>)>, SemanticError> {
    let Expr::Call(call) = expression else {
        return Ok(None);
    };
    let Expr::Member(member) = call.callee.as_ref() else {
        return Ok(None);
    };
    if member.member.name == "builder" {
        let Some(class) = static_class_name(&member.object, aliases) else {
            return Ok(None);
        };
        if !call.args.is_empty() {
            return Err(error(call.span, "`builder` does not accept arguments"));
        }
        return Ok(Some((class, Vec::new())));
    }

    let Some((class, mut fields)) = builder_plan(&member.object, scope, aliases)? else {
        return Ok(None);
    };
    let (field, value) = if member.member.name == "set" {
        if call.args.len() != 2 || call.args.iter().any(|argument| argument.name.is_some()) {
            return Err(error(
                call.span,
                "dynamic builder `set` expects a field name and value",
            ));
        }
        let Expr::Literal(Literal::String { value: field, .. }) = &call.args[0].value else {
            return Err(error(
                call.args[0].span,
                "dynamic builder field name must be a string literal",
            ));
        };
        (field.clone(), &call.args[1].value)
    } else {
        if call.args.len() != 1 || call.args[0].name.is_some() {
            return Err(error(
                call.span,
                "typed builder setters expect exactly one value",
            ));
        }
        (member.member.name.clone(), &call.args[0].value)
    };
    fields.push((field, value));
    Ok(Some((class, fields)))
}

fn lower_construction_fields(
    span: severian_ast::Span,
    class: &str,
    assignments: Vec<(String, &Expr)>,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<(String, Expression)>, SemanticError> {
    let fields = class_fields(class, aliases);
    let defaults = class_default_fields(class, aliases);
    let lowered = lower_update_fields(span, class, assignments, scope, signatures, aliases)?;
    let assigned = lowered
        .iter()
        .map(|(field, _)| field.as_str())
        .collect::<HashSet<_>>();
    let missing = fields
        .iter()
        .copied()
        .filter(|field| !assigned.contains(field) && !defaults.contains(field))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(error(
            span,
            format!(
                "builder for `{class}` is missing required field(s): {}",
                missing.join(", ")
            ),
        ));
    }
    Ok(lowered)
}

fn lower_update_fields(
    span: severian_ast::Span,
    class: &str,
    assignments: Vec<(String, &Expr)>,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<(String, Expression)>, SemanticError> {
    let fields = class_fields(class, aliases);
    let mut seen = HashSet::new();
    let mut lowered = Vec::new();
    for (field, value) in assignments {
        if field.is_empty() {
            return Err(error(
                span,
                "positional arguments may not follow named class fields",
            ));
        }
        if !fields.contains(&field.as_str()) {
            return Err(error(
                value.span(),
                format!("class `{class}` has no field `{field}`"),
            ));
        }
        if !seen.insert(field.clone()) {
            return Err(error(
                value.span(),
                format!("field `{field}` is assigned more than once"),
            ));
        }
        let (value, actual) = lower_expression(value, scope, signatures, aliases)?;
        if let Some(expected) = aliases
            .get(&format!("__class_field_type.{class}.{field}"))
            .and_then(|encoded| decode_field_type(encoded))
        {
            compatible(span, actual, expected)?;
        }
        lowered.push((field, value));
    }
    Ok(lowered)
}

fn validate_structural_conversion(
    span: severian_ast::Span,
    source: &str,
    target: &str,
    aliases: &HashMap<String, String>,
) -> Result<(), SemanticError> {
    let mut visited = HashSet::new();
    validate_structural_conversion_inner(span, source, target, aliases, &mut visited)
}

fn validate_structural_conversion_inner(
    span: severian_ast::Span,
    source: &str,
    target: &str,
    aliases: &HashMap<String, String>,
    visited: &mut HashSet<(String, String)>,
) -> Result<(), SemanticError> {
    if !visited.insert((source.to_owned(), target.to_owned())) {
        return Err(error(
            span,
            format!(
                "recursive structural conversion `{source}` -> `{target}` requires an explicit `From` implementation"
            ),
        ));
    }
    let source_fields = class_fields(source, aliases);
    let target_fields = class_fields(target, aliases);
    let defaults = class_default_fields(target, aliases);
    for field in target_fields {
        if !source_fields.contains(&field) {
            if defaults.contains(&field) {
                continue;
            }
            return Err(error(
                span,
                format!(
                    "cannot convert `{source}` to `{target}`: required field `{field}` is missing"
                ),
            ));
        }
        let source_type = aliases
            .get(&format!("__class_field_type.{source}.{field}"))
            .and_then(|encoded| decode_field_type(encoded));
        let target_type = aliases
            .get(&format!("__class_field_type.{target}.{field}"))
            .and_then(|encoded| decode_field_type(encoded));
        if let (Some(actual), Some(expected)) = (source_type, target_type) {
            compatible(span, actual, expected)?;
        }
        let source_class = aliases.get(&format!("__class_field_class.{source}.{field}"));
        let target_class = aliases.get(&format!("__class_field_class.{target}.{field}"));
        match (source_class, target_class) {
            (Some(source_class), Some(target_class)) if source_class != target_class => {
                validate_structural_conversion_inner(
                    span,
                    source_class,
                    target_class,
                    aliases,
                    visited,
                )?;
            }
            (None, Some(target_class)) => {
                return Err(error(
                    span,
                    format!("cannot convert `{source}.{field}` to nominal field `{target_class}`"),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn resolve_linked_function<'a>(
    imported: &'a str,
    signatures: &HashMap<String, Signature>,
) -> &'a str {
    if signatures.contains_key(imported) {
        imported
    } else {
        imported
            .rsplit_once('.')
            .map(|(_, function)| function)
            .filter(|function| signatures.contains_key(*function))
            .unwrap_or(imported)
    }
}

pub(super) fn method_return_type(object: ValueType, method: &str) -> ValueType {
    match (object, method) {
        (
            ValueType::String,
            "characters" | "words" | "split" | "rsplit" | "splitlines" | "split_lines" | "lines",
        )
        | (ValueType::List, "reversed" | "sorted" | "map" | "filter")
        | (ValueType::Map, "keys" | "values")
        | (ValueType::Set, "to_list" | "toList") => ValueType::List,
        (ValueType::String | ValueType::List, "frequencies") => ValueType::Map,
        (ValueType::List, "to_set" | "toSet") | (ValueType::Set, "difference") => ValueType::Set,
        (ValueType::String | ValueType::List, "join") => ValueType::String,
        (ValueType::String, "filter") => ValueType::String,
        (ValueType::String, "partition" | "rpartition") => ValueType::Tuple,
        (ValueType::List, "reduce") => ValueType::Any,
        (ValueType::String, "length" | "find" | "rfind" | "index" | "rindex" | "count") => {
            ValueType::Int
        }
        (
            ValueType::String
            | ValueType::List
            | ValueType::Tuple
            | ValueType::Map
            | ValueType::Set
            | ValueType::Tensor(_)
            | ValueType::TensorAny,
            "len" | "size" | "bytes" | "bits" | "capacity",
        ) => ValueType::Int,
        (
            ValueType::String,
            "starts_with" | "startsWith" | "ends_with" | "endsWith" | "contains" | "is_empty"
            | "is_space" | "is_alpha" | "is_digit" | "is_alnum" | "is_ascii" | "is_lower"
            | "is_upper" | "is_ascii_alnum" | "is_word" | "is_punctuation",
        ) => ValueType::Bool,
        (
            ValueType::String,
            "strip"
            | "lstrip"
            | "rstrip"
            | "lower"
            | "upper"
            | "capitalize"
            | "title"
            | "swapcase"
            | "collapse_space"
            | "collapse_horizontal_space"
            | "normalize_space"
            | "trim_prefix"
            | "trim_suffix"
            | "remove_prefix"
            | "remove_suffix"
            | "translate"
            | "replace_many"
            | "remove"
            | "remove_all"
            | "repeat"
            | "pad_left"
            | "pad_right"
            | "center"
            | "first"
            | "take"
            | "last"
            | "drop"
            | "slice"
            | "before"
            | "after"
            | "before_last"
            | "after_last"
            | "between"
            | "replace",
        ) => ValueType::String,
        (
            ValueType::List,
            "append" | "append_left" | "appendleft" | "extend" | "insert" | "remove" | "heap_push"
            | "heapPush",
        ) => ValueType::Unit,
        (
            ValueType::Set,
            "union" | "intersection" | "symmetric_difference" | "symmetricDifference",
        ) => ValueType::Set,
        (ValueType::Any, "set") => ValueType::Unit,
        _ => ValueType::Any,
    }
}

pub(super) fn collection_shape_mutating_method(method: &str) -> bool {
    matches!(
        method,
        "append"
            | "append_left"
            | "appendleft"
            | "extend"
            | "insert"
            | "remove"
            | "pop"
            | "pop_left"
            | "popleft"
            | "heap_push"
            | "heapPush"
            | "heap_pop"
            | "heapPop"
            | "clear"
    )
}

pub(super) fn validate_builtin_method(
    span: Span,
    object: ValueType,
    method: &str,
    args: &[ValueType],
) -> Result<(), SemanticError> {
    let arity = match (object, method) {
        (ValueType::List, "pop") => Some(0..=1),
        (ValueType::List, "sorted") => Some(0..=2),
        (ValueType::List, "reduce") => Some(1..=2),
        (
            ValueType::List,
            "append" | "append_left" | "appendleft" | "extend" | "remove" | "heap_push"
            | "heapPush" | "join" | "map" | "filter",
        ) => Some(1..=1),
        (ValueType::List, "insert") => Some(2..=2),
        (
            ValueType::List,
            "pop_left" | "popleft" | "heap_pop" | "heapPop" | "last" | "reversed" | "sum"
            | "minimum" | "maximum" | "frequencies" | "to_set" | "toSet",
        ) => Some(0..=0),
        (
            ValueType::Set,
            "union"
            | "intersection"
            | "difference"
            | "symmetric_difference"
            | "symmetricDifference",
        ) => Some(1..=1),
        (ValueType::Set, "to_list" | "toList") => Some(0..=0),
        (ValueType::Map, "get" | "set_default" | "setDefault") => Some(2..=2),
        (ValueType::Map, "keys" | "values") => Some(0..=0),
        (
            ValueType::String,
            "characters"
            | "words"
            | "splitlines"
            | "split_lines"
            | "lines"
            | "frequencies"
            | "strip"
            | "lstrip"
            | "rstrip"
            | "lower"
            | "upper"
            | "capitalize"
            | "title"
            | "swapcase"
            | "collapse_space"
            | "collapse_horizontal_space"
            | "normalize_space"
            | "is_empty"
            | "is_space"
            | "is_alpha"
            | "is_digit"
            | "is_alnum"
            | "is_ascii"
            | "is_lower"
            | "is_upper"
            | "is_ascii_alnum"
            | "is_word"
            | "is_punctuation"
            | "length",
        ) => Some(0..=0),
        (
            ValueType::String,
            "starts_with" | "startsWith" | "ends_with" | "endsWith" | "contains" | "find" | "rfind"
            | "index" | "rindex" | "count" | "trim_prefix" | "trim_suffix" | "remove_prefix"
            | "remove_suffix" | "translate" | "replace_many" | "remove" | "remove_all" | "repeat"
            | "pad_left" | "pad_right" | "center" | "first" | "take" | "last" | "drop" | "before"
            | "after" | "before_last" | "after_last" | "join" | "filter",
        ) => Some(1..=1),
        (ValueType::String, "slice" | "between") => Some(2..=2),
        (ValueType::String, "split") => Some(0..=2),
        (ValueType::String, "rsplit") => Some(1..=2),
        (ValueType::String, "partition" | "rpartition") => Some(1..=1),
        (ValueType::String, "replace") => Some(2..=3),
        (
            ValueType::String
            | ValueType::List
            | ValueType::Tuple
            | ValueType::Map
            | ValueType::Set
            | ValueType::Tensor(_)
            | ValueType::TensorAny,
            "len" | "size" | "bytes" | "bits" | "capacity",
        ) => Some(0..=0),
        (ValueType::Any, "get") => Some(1..=1),
        (ValueType::Any, "set") => Some(2..=2),
        _ => None,
    };
    if let Some(arity) = arity {
        if !arity.contains(&args.len()) {
            return Err(error(
                span,
                format!("method `{method}` received an invalid number of arguments"),
            ));
        }
    }
    if object == ValueType::List && matches!(method, "map" | "filter" | "reduce") {
        if !matches!(args.first(), Some(ValueType::Function | ValueType::Any)) {
            return Err(error(span, format!("method `{method}` expects a callable")));
        }
    }
    if object == ValueType::String && method == "filter" {
        if !matches!(args.first(), Some(ValueType::Function | ValueType::Any)) {
            return Err(error(span, "method `filter` expects a callable"));
        }
    }
    if object == ValueType::String && method == "remove" {
        if !matches!(args.first(), Some(ValueType::String | ValueType::List)) {
            return Err(error(
                span,
                "method `remove` expects a string of characters or a list of exact strings",
            ));
        }
    }
    if object == ValueType::List && method == "sorted" && !args.is_empty() {
        if !matches!(
            args[0],
            ValueType::Bool | ValueType::Function | ValueType::Any
        ) {
            return Err(error(
                span,
                "method `sorted` expects a reverse flag or key callable",
            ));
        }
        if args.len() == 2 && args[1] != ValueType::Bool && args[1] != ValueType::Any {
            return Err(error(
                span,
                "method `sorted` expects a boolean reverse flag",
            ));
        }
    }
    if object == ValueType::Any && matches!(method, "get" | "set") {
        if !matches!(args.first(), Some(ValueType::String | ValueType::Any)) {
            return Err(error(
                span,
                format!("object.{method} expects a string field name"),
            ));
        }
    }
    Ok(())
}
