use crate::TypeDefinitionId;

/// A stable reference to a declaration loaded from `core.primitives`.
///
/// Primitive identity is declaration identity. This type intentionally does
/// not enumerate the primitive names exposed by the language library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveId(pub TypeDefinitionId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveCategory {
    Boolean,
    Integer,
    Float,
    Text,
    Bytes,
    Absence,
    Unit,
}

/// Compiler-readable facts copied from a primitive declaration during
/// bootstrap. Names remain useful for diagnostics, but semantic decisions use
/// these facts and the declaration-backed id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveDefinition {
    pub id: PrimitiveId,
    pub name: String,
    pub category: PrimitiveCategory,
    pub representation: String,
    pub bit_width: Option<u16>,
    pub signed: Option<bool>,
    pub default_literal: bool,
}

impl PrimitiveDefinition {
    pub const fn is_numeric(&self) -> bool {
        matches!(
            self.category,
            PrimitiveCategory::Integer | PrimitiveCategory::Float
        )
    }

    pub const fn is_ordered(&self) -> bool {
        matches!(
            self.category,
            PrimitiveCategory::Integer
                | PrimitiveCategory::Float
                | PrimitiveCategory::Text
                | PrimitiveCategory::Bytes
        )
    }
}
