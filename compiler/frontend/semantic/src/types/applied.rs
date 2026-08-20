use crate::analyzer::*;

/// Resolved application of a named declaration. Primitive identity appears
/// only in `TypeKind::Primitive`; applied library types remain ordinary named
/// types and cannot accidentally enter the primitive path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::analyzer) struct AppliedType {
    pub base: TypeDefinitionId,
    pub arguments: Vec<TypeId>,
}
