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
                .ok_or_else(|| error(name.span, format!("unknown argument `{}`", name.name)))?
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
                    format!("missing argument `{}`", param.name),
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let external_operation = aliases.contains_key(&format!("__external_function.{function}"));
    let linked_function = if external_operation {
        function
    } else {
        function
            .rsplit_once('.')
            .map(|(_, name)| name)
            .filter(|name| signatures.contains_key(*name))
            .unwrap_or(function)
    };
    let runtime_function = match linked_function {
        // The MLIR backend currently exposes this intrinsic under its C symbol.
        // Its type comes from library/math, not from this mapping.
        "math.sqrt" => "sqrt",
        _ => linked_function,
    };
    Ok((
        Expression::Call {
            target: if runtime_function == linked_function {
                signature.target.clone()
            } else {
                CallTarget::source(runtime_function)
            },
            args,
        },
        instantiate_signature_type(&signature.returns, &tensor_types),
    ))
}

fn compatible_signature(
    span: Span,
    actual: ValueType,
    expected: &SignatureType,
    bindings: &mut HashMap<String, TensorElementType>,
) -> Result<(), SemanticError> {
    let SignatureType::TensorGeneric(generic) = expected else {
        return compatible(span, actual, expected.erased());
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
            Ok(SignatureParameter {
                name: param.name.name.clone(),
                ty: param.ty.as_ref().map_or_else(
                    || Ok(SignatureType::Concrete(ValueType::Any)),
                    |ty| lower_signature_type(ty, &generic_params),
                )?,
                function_return: function_return_type(param.ty.as_ref()),
                default: param.default.clone(),
            })
        })
        .collect::<Result<Vec<_>, SemanticError>>()?;
    let returns = return_type.map_or_else(
        || Ok(SignatureType::Concrete(ValueType::Unit)),
        |ty| lower_signature_type(ty, &generic_params),
    )?;
    let target = match native_symbol {
        Some(symbol) => CallTarget::native(name, symbol),
        None => CallTarget::source(name),
    }
    .with_signature(
        params.iter().map(|param| param.ty.erased()),
        returns.erased(),
    );
    Ok(Signature {
        target,
        params,
        returns,
    })
}

fn tensor_constraint(ty: &Type) -> Result<severian_hir::TensorElementConstraint, SemanticError> {
    let Type::Named(path) = ty else {
        return Err(error(ty.span(), "tensor dtype constraint must be a named capability"));
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
        return lower_type(ty).map(SignatureType::Concrete);
    }
    let Some(Type::Named(element)) = path.args.first().and_then(TypeArg::as_type) else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let Some(variable) = element.segments.first().map(|segment| segment.name.as_str()) else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let Some(constraints) = generics.get(variable) else {
        return lower_type(ty).map(SignatureType::Concrete);
    };
    let mut dimensions = [TensorDimension::Dynamic; 8];
    if path.args.len().saturating_sub(1) > dimensions.len() {
        return Err(error(path.span, "tensor rank exceeds the supported maximum of 8"));
    }
    for (axis, argument) in path.args[1..].iter().enumerate() {
        dimensions[axis] = match argument {
            TypeArg::Dimension { size, .. } => TensorDimension::Static(*size),
            TypeArg::Type { ty, .. }
                if matches!(ty.as_ref(), Type::Named(name) if name.segments.first().is_some_and(|part| part.name == "dynamic")) =>
            {
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
