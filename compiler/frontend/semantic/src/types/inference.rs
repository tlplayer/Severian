use crate::analyzer::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::analyzer) enum LiteralClass {
    Boolean,
    Integer,
    Float,
    String,
    None,
    Unit,
}

pub(in crate::analyzer) fn infer_literal_primitive(
    literal: LiteralClass,
    primitives: &PrimitiveCatalog,
) -> Result<PrimitiveId, SemanticError> {
    let category = match literal {
        LiteralClass::Boolean => PrimitiveCategory::Boolean,
        LiteralClass::Integer => PrimitiveCategory::Integer,
        LiteralClass::Float => PrimitiveCategory::Float,
        LiteralClass::String => PrimitiveCategory::Text,
        LiteralClass::None => PrimitiveCategory::Absence,
        LiteralClass::Unit => PrimitiveCategory::Unit,
    };
    primitives.default_for(category).ok_or_else(|| {
        error(
            Span::dummy(),
            format!(
                "core primitive bootstrap has no default literal declaration for `{category:?}`"
            ),
        )
    })
}
