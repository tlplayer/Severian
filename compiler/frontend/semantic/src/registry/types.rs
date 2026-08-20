use super::*;

pub(in crate::analyzer) const COMPILER_FUNCTION_NAMES: &[&str] = &[
    "abs",
    "all",
    "any",
    "bits",
    "bytes",
    "capacity",
    "divmod",
    "enumerate",
    "float",
    "indices",
    "int",
    "len",
    "max",
    "min",
    "panic",
    "print",
    "range",
    "registry",
    "size",
    "sqrt",
    "string",
    "zip",
];

pub(in crate::analyzer) fn is_compiler_function_name(name: &str) -> bool {
    COMPILER_FUNCTION_NAMES.contains(&name)
}

pub(in crate::analyzer) fn validate_no_explicit_self_parameter(
    parameters: &[severian_ast::Parameter],
) -> Result<(), SemanticError> {
    let Some(parameter) = parameters
        .first()
        .filter(|parameter| parameter.name.name == "self")
    else {
        return Ok(());
    };
    Err(error(
        parameter.name.span,
        "E000209: `self` is an implicit class receiver and must not be declared as a parameter",
    ))
}
pub(in crate::analyzer) fn class_type_name(
    ty: &Type,
    primitives: &PrimitiveCatalog,
) -> Option<String> {
    let Type::Named(path) = ty else {
        return None;
    };

    let name = path.segments.last()?.name.as_str();

    if primitives.contains(name) {
        return None;
    }

    if matches!(
        name,
        "list"
            | "map"
            | "set"
            | "Tensor"
            | "Channel"
            | "Function"
    ) {
        return None;
    }

    Some(if path.args.is_empty() {
        name.to_owned()
    } else {
        declaration_type_key(ty)
    })
}

pub(in crate::analyzer) fn resolved_class_type_name(
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
pub(in crate::analyzer) fn lower_type(
    ty: &Type,
    primitives: &PrimitiveCatalog,
) -> Result<ValueType, SemanticError> {
    match ty {
        Type::Union { .. } => Ok(ValueType::Any),

        Type::Named(path) => {
            let name = path
                .segments
                .last()
                .map(|segment| segment.name.as_str())
                .unwrap_or("");

            if let Some(ty) = primitives.value_type(name) {
                return Ok(ty);
            }

            match name {
                "list" => Ok(ValueType::List),
                "map" => Ok(ValueType::Map),
                "set" => Ok(ValueType::Set),
                "Tensor" => Ok(ValueType::Tensor(
                    lower_tensor_type(path)?
                )),
                "Channel" => Ok(ValueType::Channel),
                "Function" => Ok(ValueType::Function),

                _ => Ok(ValueType::Any),
            }
        }

        _ => Err(error(
            ty.span(),
            "type is not supported yet",
        )),
    }
}

pub(in crate::analyzer) fn declared_value_type(
    ty: &Type,
    aliases: &HashMap<String, String>,
    primitives: &PrimitiveCatalog,
) -> ValueType {
    let Some(name) = class_type_name(ty, primitives) else {
        return lower_type(ty, primitives)
            .unwrap_or(ValueType::Any);
    };

    if aliases.contains_key(&format!("__trait.{name}")) {
        ValueType::Interface(
            TypeDefinitionId::from_name(
                &canonical_type_identity(ty, aliases),
            ),
        )
    } else {
        lower_type(ty, primitives)
            .unwrap_or(ValueType::Any)
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

pub(in crate::analyzer) fn lower_tensor_type(
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

pub(in crate::analyzer) fn always_returns(instructions: &[Instruction]) -> bool {
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

pub(in crate::analyzer) fn is_upper_camel_case(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

pub(in crate::analyzer) fn value_span(value: &Option<Expr>) -> Span {
    value.as_ref().map_or(Span::dummy(), Expr::span)
}
