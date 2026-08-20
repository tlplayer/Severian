use crate::analyzer::*;

/// Primitive assignability is structural. Expression analysis must not grow a
/// primitive-name match table.
pub(in crate::analyzer) fn primitive_assignable(
    actual: &PrimitiveDefinition,
    expected: &PrimitiveDefinition,
) -> bool {
    if actual.id == expected.id {
        return true;
    }
    match (actual.category, expected.category) {
        (PrimitiveCategory::Integer, PrimitiveCategory::Integer) => {
            actual.signed == expected.signed
                && width_fits(actual.bit_width, expected.bit_width)
        }
        (PrimitiveCategory::Float, PrimitiveCategory::Float) => {
            width_fits(actual.bit_width, expected.bit_width)
        }
        _ => false,
    }
}

pub(in crate::analyzer) fn primitive_arithmetic_result(
    left: &PrimitiveDefinition,
    right: &PrimitiveDefinition,
) -> Option<PrimitiveId> {
    (left.id == right.id && left.is_numeric()).then_some(left.id)
}

pub(in crate::analyzer) fn primitive_equality_allowed(
    left: &PrimitiveDefinition,
    right: &PrimitiveDefinition,
) -> bool {
    left.id == right.id
}

pub(in crate::analyzer) fn primitive_ordering_allowed(
    left: &PrimitiveDefinition,
    right: &PrimitiveDefinition,
) -> bool {
    left.id == right.id && left.is_ordered()
}

fn width_fits(actual: Option<u16>, expected: Option<u16>) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => actual <= expected,
        (None, None) => true,
        _ => false,
    }
}

/// Transitional compatibility for execution-oriented `ValueType` users. It is
/// deliberately category-based; exact primitive checks use the functions
/// above and the declaration-backed identities retained in HIR metadata.
pub(in crate::analyzer) fn merge_numeric(
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

pub(in crate::analyzer) fn power_type(
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

pub(in crate::analyzer) fn compatible(
    span: Span,
    actual: ValueType,
    expected: ValueType,
) -> Result<(), SemanticError> {
    if actual == expected
        || actual == ValueType::Any
        || expected == ValueType::Any
        || matches!((actual, expected), (ValueType::Tensor(_), ValueType::TensorAny))
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

pub(in crate::analyzer) fn value_type_name(ty: ValueType) -> String {
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
