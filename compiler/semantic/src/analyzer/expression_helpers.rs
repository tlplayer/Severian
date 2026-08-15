use super::*;

pub(super) fn propagate_unknown_collection_shapes(
    outer: &mut HashMap<String, Binding>,
    inner: &HashMap<String, Binding>,
) {
    for (name, binding) in outer.iter_mut() {
        if binding.collection_len.is_some()
            && inner
                .get(name)
                .is_some_and(|inner| inner.collection_len.is_none())
        {
            binding.collection_len = None;
        }
    }
}

pub(super) fn lower_declared_call(
    call: &severian_ast::CallExpr,
    function: &str,
    signature: &Signature,
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<(Expression, ValueType), SemanticError> {
    let mut supplied: Vec<Option<Expression>> = vec![None; signature.params.len()];
    let mut tensor_types = HashMap::new();
    let mut positional = 0;
    for argument in &call.args {
        let index = if let Some(name) = &argument.name {
            signature
                .params
                .iter()
                .position(|param| param.name == name.name)
                .ok_or_else(|| {
                    let suggestion = closest_parameter(&name.name, &signature.params);
                    let message = suggestion.map_or_else(
                        || format!("E000204: unknown argument `{}`", name.name),
                        |suggestion| {
                            format!(
                                "E000204: unknown argument `{}`; did you mean `{suggestion}`?",
                                name.name
                            )
                        },
                    );
                    error(name.span, message)
                })?
        } else {
            let index = positional;
            positional += 1;
            index
        };
        if index >= supplied.len() || supplied[index].is_some() {
            return Err(error(
                argument.span,
                format!("invalid arguments for `{function}`"),
            ));
        }
        let (value, ty) = lower_expression(&argument.value, scope, signatures, aliases)?;
        compatible_signature(
            argument.span,
            ty,
            &signature.params[index].ty,
            expression_class(&argument.value, scope, aliases).as_deref(),
            aliases,
            &mut tensor_types,
        )?;
        supplied[index] = Some(value);
    }
    let args = supplied
        .into_iter()
        .zip(&signature.params)
        .map(|(value, param)| {
            if let Some(value) = value {
                Ok(value)
            } else if let Some(default) = &param.default {
                lower_expression(default, scope, signatures, aliases).map(|(value, _)| value)
            } else {
                Err(error(
                    call.span,
                    format!(
                        "E000203: missing argument `{}`; expected `{}`",
                        param.name,
                        value_type_name(param.ty.resolved(aliases))
                    ),
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let return_type = instantiate_signature_type(&signature.returns, &tensor_types, aliases);
    let return_type = tensor::infer_call_result(&signature.target, &args, return_type, call.span)?;
    Ok((
        Expression::Call {
            target: signature.target.clone(),
            args,
        },
        return_type,
    ))
}

fn closest_parameter<'a>(name: &str, params: &'a [SignatureParameter]) -> Option<&'a str> {
    let mut candidates = params
        .iter()
        .map(|parameter| {
            (
                edit_distance(name, &parameter.name),
                parameter.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    let (distance, candidate) = *candidates.first()?;
    let threshold = 2.max(name.chars().count() / 3);
    (distance <= threshold
        && candidates
            .get(1)
            .is_none_or(|(next_distance, _)| *next_distance > distance))
    .then_some(candidate)
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut current = vec![left_index + 1];
        for (right_index, right_character) in right.iter().enumerate() {
            current.push(
                (previous[right_index + 1] + 1)
                    .min(current[right_index] + 1)
                    .min(previous[right_index] + usize::from(left_character != *right_character)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

fn compatible_signature(
    span: Span,
    actual: ValueType,
    expected: &SignatureType,
    actual_class: Option<&str>,
    aliases: &HashMap<String, String>,
    bindings: &mut HashMap<String, TensorElementType>,
) -> Result<(), SemanticError> {
    let generic = match expected {
        SignatureType::Concrete(expected) => return compatible(span, actual, *expected),
        SignatureType::Declared(expected) => {
            return compatible_declared_type(span, actual_class, expected, aliases)
        }
        SignatureType::TensorGeneric(generic) => generic,
    };
    if actual == ValueType::Any || actual == ValueType::TensorAny {
        return Ok(());
    }
    let ValueType::Tensor(actual) = actual else {
        return Err(error(span, "generic tensor parameter requires a tensor"));
    };
    if !generic
        .constraints
        .iter()
        .all(|constraint| actual.element.satisfies(*constraint))
    {
        return Err(error(
            span,
            format!(
                "tensor element `{}` does not satisfy the constraints on `{}`",
                actual.element.name(),
                generic.variable
            ),
        ));
    }
    if let Some(bound) = bindings.get(&generic.variable) {
        if *bound != actual.element {
            return Err(error(
                span,
                format!(
                    "generic tensor `{}` was bound to `{}`, then used with `{}`",
                    generic.variable,
                    bound.name(),
                    actual.element.name()
                ),
            ));
        }
    } else {
        bindings.insert(generic.variable.clone(), actual.element);
    }
    let expected_shape = TensorType {
        element: actual.element,
        rank: generic.rank,
        dimensions: generic.dimensions,
    };
    if actual.is_compatible_with(expected_shape) {
        Ok(())
    } else {
        Err(error(span, "tensor rank or dimensions do not match"))
    }
}

fn instantiate_signature_type(
    ty: &SignatureType,
    bindings: &HashMap<String, TensorElementType>,
    aliases: &HashMap<String, String>,
) -> ValueType {
    match ty {
        SignatureType::Concrete(ty) => *ty,
        SignatureType::TensorGeneric(generic) => {
            let Some(element) = bindings.get(&generic.variable).copied() else {
                return ValueType::TensorAny;
            };
            ValueType::Tensor(TensorType {
                element,
                rank: generic.rank,
                dimensions: generic.dimensions,
            })
        }
        SignatureType::Declared(ty) => declared_value_type(ty, aliases),
    }
}

pub(super) fn collection_length(expression: &Expr) -> Option<usize> {
    match expression {
        Expr::List(collection) | Expr::Tuple(collection) => Some(collection.elements.len()),
        _ => None,
    }
}

pub(super) fn constant_integer(expression: &Expr) -> Option<i64> {
    match expression {
        Expr::Literal(Literal::Integer { value, .. }) => Some(*value),
        Expr::Binary(binary) => {
            let left = constant_integer(&binary.left)?;
            let right = constant_integer(&binary.right)?;
            match binary.op {
                AstBinaryOp::Add => left.checked_add(right),
                AstBinaryOp::Sub => left.checked_sub(right),
                AstBinaryOp::Mul => left.checked_mul(right),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn named_type_is(ty: &Type, expected: &str) -> bool {
    matches!(ty, Type::Named(path) if path.segments.first().is_some_and(|segment| segment.name == expected))
}

pub(super) fn checked_integer_overflow(
    expression: &Expr,
    scope: &HashMap<String, Binding>,
) -> bool {
    let Expr::Binary(binary) = expression else {
        return false;
    };
    let (binding, constant, operation) = match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Identifier(identifier), right) => (
            scope.get(&identifier.name),
            constant_integer(right),
            binary.op,
        ),
        _ => return false,
    };
    let (Some(constant), Some(value), Some(maximum)) = (
        constant,
        binding.and_then(|binding| binding.known_integer),
        binding.and_then(|binding| binding.integer_max),
    ) else {
        return false;
    };
    let result = match operation {
        AstBinaryOp::Add => value.checked_add(constant),
        AstBinaryOp::Sub => value.checked_sub(constant),
        AstBinaryOp::Mul => value.checked_mul(constant),
        _ => return false,
    };
    result.is_none_or(|result| !(0..=maximum).contains(&result))
}

pub(super) fn inclusive_collection_range(expression: &Expr) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        return false;
    };
    if callee.name != "range" || call.args.len() != 2 {
        return false;
    }
    let Expr::Binary(end) = &call.args[1].value else {
        return false;
    };
    if end.op != AstBinaryOp::Add || constant_integer(&end.right) != Some(1) {
        return false;
    }
    let Expr::Call(size) = end.left.as_ref() else {
        return false;
    };
    matches!(size.callee.as_ref(), Expr::Identifier(name) if name.name == "size")
}

pub(super) fn lower_format_args(
    template: &str,
    scope: &HashMap<String, Binding>,
    span: Span,
) -> Result<(Vec<Expression>, Vec<ValueType>), SemanticError> {
    let mut args = Vec::new();
    let mut arg_types = Vec::new();
    let mut remainder = template;

    while let Some(open) = remainder.find('{') {
        remainder = &remainder[open + 1..];
        let close = remainder
            .find('}')
            .ok_or_else(|| error(span, "formatted string has an unmatched `{`"))?;
        let name = &remainder[..close];
        if name.is_empty()
            || !name.bytes().enumerate().all(|(index, byte)| {
                byte == b'_'
                    || byte.is_ascii_alphanumeric() && (index > 0 || !byte.is_ascii_digit())
            })
        {
            return Err(error(
                span,
                format!("unsupported formatted string field `{{{name}}}`"),
            ));
        }
        let binding = scope
            .get(name)
            .ok_or_else(|| error(span, format!("unknown formatted string field `{name}`")))?;
        args.push(Expression::Variable(name.into()));
        arg_types.push(binding.ty);
        remainder = &remainder[close + 1..];
    }

    if remainder.contains('}') {
        return Err(error(span, "formatted string has an unmatched `}`"));
    }
    Ok((args, arg_types))
}

pub(super) fn lower_comprehension_clauses(
    clauses: &[severian_ast::ComprehensionClause],
    scope: &mut HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<HirComprehensionClause>, SemanticError> {
    let mut lowered = Vec::new();
    for clause in clauses {
        let iterable = lower_expression(&clause.iterable, scope, signatures, aliases)?.0;
        let pattern = lower_pattern(&clause.pattern, scope, aliases)?;
        let condition = clause
            .condition
            .as_ref()
            .map(|condition| {
                let (condition, ty) = lower_expression(condition, scope, signatures, aliases)?;
                compatible(clause.iterable.span(), ty, ValueType::Bool)?;
                Ok(condition)
            })
            .transpose()?;
        lowered.push(HirComprehensionClause {
            pattern,
            iterable,
            condition,
        });
    }
    Ok(lowered)
}

pub(super) fn lower_collection(
    elements: &[Expr],
    scope: &HashMap<String, Binding>,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Result<Vec<Expression>, SemanticError> {
    elements
        .iter()
        .map(|element| {
            lower_expression(element, scope, signatures, aliases).map(|(element, _)| element)
        })
        .collect()
}

pub(super) fn lower_signature(
    name: &str,
    native_symbol: Option<&str>,
    generic_params: &[severian_ast::GenericParameter],
    params: &[severian_ast::Parameter],
    return_type: Option<&Type>,
    aliases: &HashMap<String, String>,
) -> Result<Signature, SemanticError> {
    let generic_params = generic_params
        .iter()
        .map(|parameter| {
            let constraints = parameter
                .constraints
                .iter()
                .map(tensor_constraint)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((parameter.name.name.clone(), constraints))
        })
        .collect::<Result<HashMap<_, _>, SemanticError>>()?;
    let params = params
        .iter()
        .map(|param| {
            let ty = param.ty.as_ref().map_or_else(
                || Ok(SignatureType::Concrete(ValueType::Any)),
                |ty| lower_signature_type(ty, &generic_params),
            )?;
            Ok(SignatureParameter {
                name: param.name.name.clone(),
                any_origin: signature_any_origin(&ty, param.ty.is_some(), AnyOrigin::Explicit),
                ty,
                function_return: function_return_type(param.ty.as_ref()),
                default: param.default.clone(),
            })
        })
        .collect::<Result<Vec<_>, SemanticError>>()?;
    let returns = return_type.map_or_else(
        || Ok(SignatureType::Concrete(ValueType::Unit)),
        |ty| lower_signature_type(ty, &generic_params),
    )?;
    let result_ok = result_ok_type(return_type)
        .map(|ty| lower_signature_type(ty, &generic_params))
        .transpose()?;
    let return_any_origin = signature_any_origin(
        &returns,
        return_type.is_some(),
        AnyOrigin::UnresolvedGeneric,
    );
    let target = match native_symbol {
        Some(symbol) => CallTarget::native(name, symbol),
        None => CallTarget::source(name),
    }
    .with_signature(
        params.iter().map(|param| param.ty.resolved(aliases)),
        returns.resolved(aliases),
    )
    .with_signature_origins(
        params.iter().map(|param| param.any_origin).collect(),
        return_any_origin,
    );
    Ok(Signature {
        target,
        params,
        returns,
        result_ok,
    })
}

fn result_ok_type(ty: Option<&Type>) -> Option<&Type> {
    match ty? {
        Type::Result { ok, .. } => Some(ok),
        Type::Named(path)
            if path
                .segments
                .first()
                .is_some_and(|segment| segment.name == "Result") =>
        {
            path.args.first().and_then(TypeArg::as_type)
        }
        _ => None,
    }
}

pub(super) fn result_payload_type(
    expression: &Expr,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Option<ValueType> {
    let Expr::Call(call) = expression else {
        return None;
    };
    let function = called_function_name(call.callee.as_ref(), aliases)?;
    signatures
        .get(&function)
        .or_else(|| signatures.get(function.rsplit('.').next()?))?
        .result_ok
        .as_ref()
        .map(|ty| ty.resolved(aliases))
}

pub(super) fn result_payload_receiver(
    expression: &Expr,
    signatures: &HashMap<String, Signature>,
    aliases: &HashMap<String, String>,
) -> Option<ReceiverType> {
    let Expr::Call(call) = expression else {
        return None;
    };
    let function = called_function_name(call.callee.as_ref(), aliases)?;
    let payload = signatures
        .get(&function)
        .or_else(|| signatures.get(function.rsplit('.').next()?))?
        .result_ok
        .as_ref()?;
    let SignatureType::Declared(ty) = payload else {
        return None;
    };
    declared_receiver_type(ty, aliases)
}

fn signature_any_origin(
    ty: &SignatureType,
    explicitly_declared: bool,
    generic_origin: AnyOrigin,
) -> Option<AnyOrigin> {
    match ty {
        SignatureType::TensorGeneric(_) => Some(generic_origin),
        SignatureType::Declared(_) => None,
        SignatureType::Concrete(ValueType::Any | ValueType::TensorAny) => {
            Some(if explicitly_declared {
                AnyOrigin::Explicit
            } else {
                AnyOrigin::InferenceFallback
            })
        }
        SignatureType::Concrete(_) => None,
    }
}

fn tensor_constraint(ty: &Type) -> Result<severian_hir::TensorElementConstraint, SemanticError> {
    let Type::Named(path) = ty else {
        return Err(error(
            ty.span(),
            "tensor dtype constraint must be a named capability",
        ));
    };
    let name = path
        .segments
        .first()
        .map(|segment| segment.name.as_str())
        .unwrap_or("");
    use severian_hir::TensorElementConstraint as Constraint;
    match name {
        "DType" | "dtype" | "Any" | "any" => Ok(Constraint::Any),
        "Numeric" | "numeric" => Ok(Constraint::Numeric),
        "Integer" | "integer" => Ok(Constraint::Integer),
        "SignedInteger" | "signed_integer" => Ok(Constraint::SignedInteger),
        "UnsignedInteger" | "unsigned_integer" => Ok(Constraint::UnsignedInteger),
        "Float" | "float" => Ok(Constraint::Float),
        "Complex" | "complex" => Ok(Constraint::Complex),
        _ => Err(error(
            ty.span(),
            format!("unknown tensor dtype constraint `{name}`"),
        )),
    }
}

fn lower_signature_type(
    ty: &Type,
    generics: &HashMap<String, Vec<severian_hir::TensorElementConstraint>>,
) -> Result<SignatureType, SemanticError> {
    let Type::Named(path) = ty else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    if !path
        .segments
        .first()
        .is_some_and(|segment| segment.name == "Tensor")
    {
        let lowered = lower_type(ty)?;
        let generic_variable =
            path.segments.len() == 1 && generics.contains_key(path.segments[0].name.as_str());
        let explicitly_dynamic =
            path.segments.len() == 1 && matches!(path.segments[0].name.as_str(), "Any" | "any");
        return Ok(
            if lowered == ValueType::Any && !explicitly_dynamic && !generic_variable {
                SignatureType::Declared(ty.clone())
            } else {
                SignatureType::Concrete(lowered)
            },
        );
    }
    let Some(Type::Named(element)) = path.args.first().and_then(TypeArg::as_type) else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let Some(variable) = element
        .segments
        .first()
        .map(|segment| segment.name.as_str())
    else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let Some(constraints) = generics.get(variable) else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let mut dimensions = [TensorDimension::Dynamic; 8];
    if path.args.len().saturating_sub(1) > dimensions.len() {
        return Err(error(
            path.span,
            "tensor rank exceeds the supported maximum of 8",
        ));
    }
    for (axis, argument) in path.args[1..].iter().enumerate() {
        dimensions[axis] = match argument {
            TypeArg::Dimension { size, .. } => TensorDimension::Static(*size),
            TypeArg::Type { ty, .. } if matches!(ty.as_ref(), Type::Named(name) if name.segments.first().is_some_and(|part| part.name == "dynamic")) => {
                TensorDimension::Dynamic
            }
            _ => {
                return Err(error(
                    argument.span(),
                    "tensor dimensions must be integers or `dynamic`",
                ))
            }
        };
    }
    Ok(SignatureType::TensorGeneric(GenericTensorType {
        variable: variable.to_string(),
        constraints: constraints.clone(),
        rank: (path.args.len() > 1).then_some((path.args.len() - 1) as u8),
        dimensions,
    }))
}

fn compatible_declared_type(
    span: Span,
    actual_class: Option<&str>,
    expected: &Type,
    aliases: &HashMap<String, String>,
) -> Result<(), SemanticError> {
    let expected_name = declaration_type_name(expected).unwrap_or_default();
    let short_name = expected_name.rsplit('.').next().unwrap_or(&expected_name);
    let resolved_expected =
        resolved_class_type_name(expected, aliases).unwrap_or_else(|| expected_name.clone());
    if aliases.contains_key(&format!("__trait.{short_name}")) {
        let Some(actual_class) = actual_class else {
            return Err(error(
                span,
                format!(
                    "expected `{}`, found a non-class value",
                    declaration_type_key(expected)
                ),
            ));
        };
        let actual_name = actual_class
            .split_once('[')
            .map_or(actual_class, |(name, _)| name)
            .rsplit('.')
            .next()
            .unwrap_or(actual_class);
        if actual_name == short_name {
            return Ok(());
        }
        if structurally_conforms(actual_class, expected, short_name, aliases) {
            return Ok(());
        }
        return Err(error(
            span,
            format!(
                "class `{actual_class}` does not structurally satisfy `{}`",
                declaration_type_key(expected)
            ),
        ));
    }
    let Some(actual_class) = actual_class else {
        // Ordinary class values still use the legacy erased runtime carrier.
        // Structural traits are the boundary where a concrete method set must
        // be statically proven, so incomplete class-flow inference remains
        // permissive here until object types have a first-class HIR carrier.
        return Ok(());
    };
    if actual_class == short_name
        || actual_class == expected_name
        || actual_class == resolved_expected
    {
        Ok(())
    } else {
        Err(error(
            span,
            format!("expected class `{short_name}`, found `{actual_class}`"),
        ))
    }
}

fn structurally_conforms(
    actual_class: &str,
    expected: &Type,
    trait_name: &str,
    aliases: &HashMap<String, String>,
) -> bool {
    let Type::Named(path) = expected else {
        return false;
    };
    let parameters = aliases
        .get(&format!("__trait_generic_params.{trait_name}"))
        .map(|parameters| {
            parameters
                .split(',')
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let arguments = path
        .args
        .iter()
        .map(|argument| argument.as_type().map(declaration_type_key))
        .collect::<Option<Vec<_>>>();
    let Some(arguments) = arguments else {
        return false;
    };
    if parameters.len() != arguments.len() {
        return false;
    }
    let substitutions = parameters
        .into_iter()
        .zip(arguments)
        .collect::<HashMap<_, _>>();
    let required_methods = aliases
        .get(&format!("__class_methods.{trait_name}"))
        .map(|methods| {
            methods
                .split(',')
                .filter(|method| !method.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let actual_methods = aliases
        .get(&format!("__class_methods.{actual_class}"))
        .map(|methods| methods.split(',').collect::<HashSet<_>>())
        .unwrap_or_default();
    required_methods.into_iter().all(|method| {
        if !actual_methods.contains(method) {
            return false;
        }
        let Some(required) =
            aliases.get(&format!("__trait_method_signature.{trait_name}.{method}"))
        else {
            return false;
        };
        let required = substitutions
            .iter()
            .fold(required.clone(), |signature, (name, value)| {
                replace_type_name(&signature, name, value)
            });
        aliases
            .get(&format!("__class_method_signature.{actual_class}.{method}"))
            .is_some_and(|actual| actual == &required)
    })
}

fn replace_type_name(source: &str, name: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(index) = rest.find(name) {
        let before = rest[..index].chars().next_back();
        let after = rest[index + name.len()..].chars().next();
        let identifier = |character: Option<char>| {
            character.is_some_and(|character| character == '_' || character.is_alphanumeric())
        };
        if !identifier(before) && !identifier(after) {
            result.push_str(&rest[..index]);
            result.push_str(replacement);
            rest = &rest[index + name.len()..];
        } else {
            result.push_str(&rest[..index + name.len()]);
            rest = &rest[index + name.len()..];
        }
    }
    result.push_str(rest);
    result
}

pub(super) fn function_return_type(ty: Option<&Type>) -> Option<ValueType> {
    let Type::Named(path) = ty? else {
        return None;
    };
    if path.segments.first()?.name != "Function" {
        return None;
    }
    path.args
        .last()
        .and_then(TypeArg::as_type)
        .and_then(|argument| lower_type(argument).ok())
}
