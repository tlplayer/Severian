use crate::analyzer::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::analyzer) enum PrimitiveConstraint {
    Numeric,
    Integer,
    Signed,
    Ordered,
    Equality,
}

pub(in crate::analyzer) fn satisfies_primitive_constraint(
    definition: &PrimitiveDefinition,
    constraint: PrimitiveConstraint,
) -> bool {
    match constraint {
        PrimitiveConstraint::Numeric => definition.is_numeric(),
        PrimitiveConstraint::Integer => definition.category == PrimitiveCategory::Integer,
        PrimitiveConstraint::Signed => definition.signed == Some(true),
        PrimitiveConstraint::Ordered => definition.is_ordered(),
        PrimitiveConstraint::Equality => true,
    }
}
