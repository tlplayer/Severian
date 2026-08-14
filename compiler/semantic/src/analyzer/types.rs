use super::*;

pub(super) fn class_type_name(ty: &Type) -> Option<String> {
    let Type::Named(path) = ty else { return None };
    let name = path.segments.last()?.name.as_str();
    if matches!(
        name,
        "int"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "float"
            | "f32"
            | "f64"
            | "bool"
            | "string"
            | "unit"
            | "list"
            | "map"
            | "set"
            | "Tensor"
            | "Channel"
            | "Function"
            | "Result"
            | "Option"
    ) {
        None
    } else {
        Some(if path.args.is_empty() {
            name.to_owned()
        } else {
            declaration_type_key(ty)
        })
    }
}

pub(super) fn resolved_class_type_name(
    ty: &Type,
    aliases: &HashMap<String, String>,
) -> Option<String> {
    let Type::Named(path) = ty else { return None };
    let raw = path
        .segments
        .iter()
        .map(|segment| segment.name.as_str())
        .collect::<Vec<_>>()
        .join(".");
    let canonical = path
        .segments
        .split_first()
        .map(|(first, rest)| {
            std::iter::once(
                aliases
                    .get(&first.name)
                    .map(String::as_str)
                    .unwrap_or(&first.name),
            )
            .chain(rest.iter().map(|segment| segment.name.as_str()))
            .collect::<Vec<_>>()
            .join(".")
        })
        .unwrap_or_default();
    aliases
        .get(&format!("__module_class.{raw}"))
        .or_else(|| aliases.get(&format!("__module_class.{canonical}")))
        .cloned()
        .or_else(|| {
            path.segments
                .last()
                .and_then(|segment| aliases.get(&segment.name))
                .and_then(|identity| identity.strip_prefix("__class."))
                .map(str::to_owned)
        })
        .or_else(|| class_type_name(ty))
}

pub(super) fn lower_type(ty: &Type) -> Result<ValueType, SemanticError> {
    match ty {
        Type::Union { .. } => Ok(ValueType::Any),
        Type::Named(path) => {
            let name = path
                .segments
                .first()
                .map(|segment| segment.name.as_str())
                .unwrap_or("");
            match name {
                "int" | "u8" | "u16" | "u32" | "u64" | "usize" => Ok(ValueType::Int),
                "float" | "f32" | "f64" => Ok(ValueType::Float),
                "bool" => Ok(ValueType::Bool),
                "string" => Ok(ValueType::String),
                "unit" => Ok(ValueType::Unit),
                "list" => Ok(ValueType::List),
                "map" => Ok(ValueType::Map),
                "set" => Ok(ValueType::Set),
                "Tensor"
                    if matches!(
                        path.args.first().and_then(TypeArg::as_type),
                        Some(Type::Named(element))
                            if element.segments.first().is_some_and(|part| {
                                is_conventional_type_variable(&part.name)
                            })
                    ) =>
                {
                    Ok(ValueType::TensorAny)
                }
                "Tensor" => Ok(ValueType::Tensor(lower_tensor_type(path)?)),
                "Channel" => Ok(ValueType::Channel),
                "Function" => Ok(ValueType::Function),
                "Result" => Ok(ValueType::Result),
                "Option" => Ok(ValueType::Option),
                _ => Ok(ValueType::Any),
            }
        }
        _ => Err(error(ty.span(), "type is not supported yet")),
    }
}

pub(super) fn declared_value_type(ty: &Type, aliases: &HashMap<String, String>) -> ValueType {
    let Some(name) = class_type_name(ty) else {
        return lower_type(ty).unwrap_or(ValueType::Any);
    };
    if aliases.contains_key(&format!("__trait.{name}")) {
        ValueType::Interface(TypeDefinitionId::from_name(&canonical_type_identity(
            ty, aliases,
        )))
    } else {
        lower_type(ty).unwrap_or(ValueType::Any)
    }
}

fn canonical_type_identity(ty: &Type, aliases: &HashMap<String, String>) -> String {
    match ty {
        Type::Named(path) => {
            let raw = path
                .segments
                .iter()
                .map(|segment| segment.name.as_str())
                .collect::<Vec<_>>()
                .join(".");
            let mut identity = canonical_declared_type_name(&raw, aliases);
            if !path.args.is_empty() {
                let arguments = path
                    .args
                    .iter()
                    .map(|argument| match argument {
                        TypeArg::Type { ty, .. } => canonical_type_identity(ty, aliases),
                        TypeArg::Dimension { size, .. } => size.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                identity.push('[');
                identity.push_str(&arguments);
                identity.push(']');
            }
            identity
        }
        _ => declaration_type_key(ty),
    }
}

fn is_conventional_type_variable(name: &str) -> bool {
    matches!(name, "type" | "T" | "K" | "V")
}

pub(super) fn lower_tensor_type(
    path: &severian_ast::TypePath,
) -> Result<TensorType, SemanticError> {
    let element = match path.args.first().and_then(TypeArg::as_type) {
        Some(Type::Named(element)) => {
            match element
                .segments
                .first()
                .and_then(|part| TensorElementType::parse(&part.name))
            {
                Some(element) => element,
                None => return Err(error(path.span, "unsupported tensor element type")),
            }
        }
        None if path.args.is_empty() => TensorElementType::F64,
        _ => {
            return Err(error(
                path.span,
                "the first Tensor argument must be an element type",
            ))
        }
    };
    if path.args.len() <= 1 {
        return Ok(TensorType::dynamic(element));
    }
    let mut dimensions = Vec::with_capacity(path.args.len() - 1);
    for argument in &path.args[1..] {
        match argument {
            TypeArg::Dimension { size, .. } => dimensions.push(TensorDimension::Static(*size)),
            TypeArg::Type { ty, .. } if matches!(ty.as_ref(), Type::Named(name) if name.segments.first().is_some_and(|part| part.name == "dynamic")) => {
                dimensions.push(TensorDimension::Dynamic)
            }
            _ => {
                return Err(error(
                    argument.span(),
                    "tensor dimensions must be integers or `dynamic`",
                ))
            }
        }
    }
    TensorType::ranked(element, &dimensions).map_err(|message| error(path.span, message))
}

pub(super) fn merge_numeric(
    left: ValueType,
    right: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if left == ValueType::Any || right == ValueType::Any {
        return Ok(ValueType::Any);
    }
    if left == right && matches!(left, ValueType::Int | ValueType::Float | ValueType::String) {
        Ok(left)
    } else {
        Err(error(span, "operator requires matching numeric values"))
    }
}

pub(super) fn power_type(
    base: ValueType,
    exponent: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if base == ValueType::Any || exponent == ValueType::Any {
        return Ok(ValueType::Any);
    }
    if base == ValueType::Int && exponent == ValueType::Int {
        return Ok(ValueType::Int);
    }
    if matches!(base, ValueType::Int | ValueType::Float)
        && matches!(exponent, ValueType::Int | ValueType::Float)
    {
        return Ok(ValueType::Float);
    }
    Err(error(span, "power requires numeric values"))
}

pub(super) fn compatible(
    span: Span,
    actual: ValueType,
    expected: ValueType,
) -> Result<(), SemanticError> {
    if actual == expected
        || actual == ValueType::Any
        || expected == ValueType::Any
        || matches!(
            (actual, expected),
            (ValueType::Tensor(_), ValueType::TensorAny)
        )
        || matches!((actual, expected), (ValueType::Tensor(actual), ValueType::Tensor(expected)) if actual.is_compatible_with(expected))
        || (expected == ValueType::Result && actual != ValueType::Unit)
    {
        Ok(())
    } else {
        Err(error(
            span,
            format!(
                "E000202: mismatched types: expected `{}`, found `{}`",
                value_type_name(expected),
                value_type_name(actual)
            ),
        ))
    }
}

pub(super) fn value_type_name(ty: ValueType) -> String {
    match ty {
        ValueType::Int => "int".into(),
        ValueType::Float => "float".into(),
        ValueType::Bool => "bool".into(),
        ValueType::String => "string".into(),
        ValueType::Unit => "unit".into(),
        ValueType::List => "list".into(),
        ValueType::Tuple => "tuple".into(),
        ValueType::Map => "map".into(),
        ValueType::Set => "set".into(),
        ValueType::Result => "Result".into(),
        ValueType::Option => "Option".into(),
        ValueType::Interface(definition) => format!("interface#{}", definition.0),
        ValueType::TensorAny => "Tensor".into(),
        ValueType::Tensor(tensor) => {
            let mut parts = vec![tensor.element.name().to_owned()];
            if let Some(rank) = tensor.rank {
                parts.extend(tensor.dimensions[..rank as usize].iter().map(|dimension| {
                    match dimension {
                        TensorDimension::Static(value) => value.to_string(),
                        TensorDimension::Dynamic => "dynamic".into(),
                    }
                }));
            }
            format!("Tensor[{}]", parts.join(", "))
        }
        other => format!("{other:?}"),
    }
}

pub(super) fn always_returns(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instruction| match instruction {
        Instruction::Return(_) => true,
        Instruction::If {
            then_instructions,
            else_instructions,
            ..
        } => always_returns(then_instructions) && always_returns(else_instructions),
        Instruction::Switch { arms, .. } => {
            !arms.is_empty() && arms.iter().all(|arm| always_returns(&arm.instructions))
        }
        Instruction::With { instructions, .. } => always_returns(instructions),
        Instruction::While { condition, .. }
            if matches!(condition.kind(), Expression::Boolean(true)) =>
        {
            true
        }
        _ => false,
    })
}

pub(super) fn is_upper_camel_case(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

pub(super) fn value_span(value: &Option<Expr>) -> Span {
    value.as_ref().map_or(Span::dummy(), Expr::span)
}
