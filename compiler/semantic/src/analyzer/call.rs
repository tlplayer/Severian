use super::*;

pub(super) fn lower_call(
    call: &severian_ast::CallExpr,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
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
            if object.name == "int" && member.member.name == "parse" {
                let args = call
                    .args
                    .iter()
                    .map(|arg| {
                        lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((
                    Expression::Call {
                        target: CallTarget::source("int.parse"),
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
                    let args = call
                        .args
                        .iter()
                        .map(|arg| {
                            lower_expression(&arg.value, scope, signatures, aliases)
                                .map(|(arg, _)| arg)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok((
                        Expression::Construct {
                            type_id: TypeDefinitionId::from_name(class),
                            class: class.clone(),
                            args,
                        },
                        ValueType::Any,
                    ));
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
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                type_id: TypeDefinitionId::from_name(class),
                class: class.clone(),
                args,
            },
            ValueType::Any,
        ));
    }
    if let Some(class) = canonical.strip_prefix("__class.") {
        let args = call
            .args
            .iter()
            .map(|arg| lower_expression(&arg.value, scope, signatures, aliases).map(|(arg, _)| arg))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((
            Expression::Construct {
                type_id: TypeDefinitionId::from_name(class),
                class: class.into(),
                args,
            },
            ValueType::Any,
        ));
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
