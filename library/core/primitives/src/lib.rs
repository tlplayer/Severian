#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;

pub const PACKAGE_NAME: &str = "core.primitives";

/// Stable identity of a declaration in the primitive bootstrap package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveId(pub u64);

impl PrimitiveId {
    pub const fn from_declaration_id(id: u64) -> Self {
        Self(id)
    }
}

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

impl PrimitiveCategory {
    pub fn from_contract(value: &str) -> Result<Self, PrimitiveError> {
        match value {
            "boolean" => Ok(Self::Boolean),
            "integer" => Ok(Self::Integer),
            "float" => Ok(Self::Float),
            "text" => Ok(Self::Text),
            "bytes" => Ok(Self::Bytes),
            "absence" => Ok(Self::Absence),
            "unit" => Ok(Self::Unit),
            other => Err(PrimitiveError::UnknownCategory(other.to_owned())),
        }
    }
}

/// Raw metadata mechanically extracted from a declaration by the compiler's
/// source adapter. Interpretation and validation belong to this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveMetadata {
    /// Canonical declaration identity assigned by the compiler resolver.
    pub id: PrimitiveId,
    pub name: String,
    pub category: Option<String>,
    pub representation: Option<String>,
    pub bits: Option<i64>,
    pub signed: Option<bool>,
    pub default_literal: Option<bool>,
}

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
    fn interpret(metadata: PrimitiveMetadata) -> Result<Self, PrimitiveError> {
        let category_name = metadata
            .category
            .ok_or_else(|| PrimitiveError::MissingCategory(metadata.name.clone()))?;
        let category = PrimitiveCategory::from_contract(&category_name)?;
        let representation = metadata
            .representation
            .filter(|representation| !representation.is_empty())
            .ok_or_else(|| PrimitiveError::MissingRepresentation(metadata.name.clone()))?;
        let bit_width = match metadata.bits {
            None | Some(0) => None,
            Some(bits) => Some(
                u16::try_from(bits)
                    .map_err(|_| PrimitiveError::InvalidBitWidth(metadata.name.clone(), bits))?,
            ),
        };
        let signed = match category {
            PrimitiveCategory::Integer => Some(metadata.signed.unwrap_or(false)),
            _ if metadata.signed.is_some() => {
                return Err(PrimitiveError::SignedNonInteger(metadata.name))
            }
            _ => None,
        };
        Ok(Self {
            id: metadata.id,
            name: metadata.name,
            category,
            representation,
            bit_width,
            signed,
            default_literal: metadata.default_literal.unwrap_or(false),
        })
    }

    pub const fn is_numeric(&self) -> bool {
        matches!(self.category, PrimitiveCategory::Integer | PrimitiveCategory::Float)
    }

    pub const fn supports_equality(&self) -> bool {
        true
    }

    pub const fn supports_ordering(&self) -> bool {
        matches!(
            self.category,
            PrimitiveCategory::Integer
                | PrimitiveCategory::Float
                | PrimitiveCategory::Text
                | PrimitiveCategory::Bytes
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveCatalog {
    by_name: HashMap<String, PrimitiveId>,
    definitions: BTreeMap<PrimitiveId, PrimitiveDefinition>,
    defaults: HashMap<PrimitiveCategory, PrimitiveId>,
}

impl PrimitiveCatalog {
    pub fn from_metadata(
        metadata: impl IntoIterator<Item = PrimitiveMetadata>,
    ) -> Result<Self, PrimitiveError> {
        let mut catalog = Self {
            by_name: HashMap::new(),
            definitions: BTreeMap::new(),
            defaults: HashMap::new(),
        };
        for metadata in metadata {
            let definition = PrimitiveDefinition::interpret(metadata)?;
            if catalog.by_name.contains_key(&definition.name) {
                return Err(PrimitiveError::DuplicateName(definition.name));
            }
            if definition.default_literal
                && catalog
                    .defaults
                    .insert(definition.category, definition.id)
                    .is_some()
            {
                return Err(PrimitiveError::DuplicateDefault(definition.category));
            }
            catalog.by_name.insert(definition.name.clone(), definition.id);
            catalog.definitions.insert(definition.id, definition);
        }
        if catalog.definitions.is_empty() {
            return Err(PrimitiveError::EmptyCatalog);
        }
        Ok(catalog)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub fn resolve(&self, name: &str) -> Option<PrimitiveId> {
        self.by_name.get(name).copied()
    }

    pub fn definition(&self, id: PrimitiveId) -> Option<&PrimitiveDefinition> {
        self.definitions.get(&id)
    }

    pub fn definitions(&self) -> impl Iterator<Item = &PrimitiveDefinition> {
        self.definitions.values()
    }

    pub fn default_for(&self, category: PrimitiveCategory) -> Option<PrimitiveId> {
        self.defaults.get(&category).copied()
    }
}

pub fn assignable(actual: &PrimitiveDefinition, expected: &PrimitiveDefinition) -> bool {
    if actual.id == expected.id {
        return true;
    }
    match (actual.category, expected.category) {
        (PrimitiveCategory::Integer, PrimitiveCategory::Integer) => {
            actual.signed == expected.signed && width_fits(actual.bit_width, expected.bit_width)
        }
        (PrimitiveCategory::Float, PrimitiveCategory::Float) => {
            width_fits(actual.bit_width, expected.bit_width)
        }
        _ => false,
    }
}

pub fn arithmetic_result(
    left: &PrimitiveDefinition,
    right: &PrimitiveDefinition,
) -> Option<PrimitiveId> {
    (left.id == right.id && left.is_numeric()).then_some(left.id)
}

pub fn equality_allowed(left: &PrimitiveDefinition, right: &PrimitiveDefinition) -> bool {
    left.id == right.id && left.supports_equality()
}

pub fn ordering_allowed(left: &PrimitiveDefinition, right: &PrimitiveDefinition) -> bool {
    left.id == right.id && left.supports_ordering()
}

fn width_fits(actual: Option<u16>, expected: Option<u16>) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => actual <= expected,
        (None, None) => true,
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveError {
    EmptyCatalog,
    DuplicateName(String),
    DuplicateDefault(PrimitiveCategory),
    UnknownCategory(String),
    MissingCategory(String),
    MissingRepresentation(String),
    InvalidBitWidth(String, i64),
    SignedNonInteger(String),
}

impl fmt::Display for PrimitiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCatalog => formatter.write_str("core primitive catalog is empty"),
            Self::DuplicateName(name) => write!(formatter, "duplicate primitive `{name}`"),
            Self::DuplicateDefault(category) => {
                write!(formatter, "multiple default literal primitives for `{category:?}`")
            }
            Self::UnknownCategory(category) => {
                write!(formatter, "unknown primitive category `{category}`")
            }
            Self::MissingCategory(name) => {
                write!(formatter, "primitive `{name}` has no category")
            }
            Self::MissingRepresentation(name) => {
                write!(formatter, "primitive `{name}` has no representation")
            }
            Self::InvalidBitWidth(name, bits) => {
                write!(formatter, "primitive `{name}` has invalid bit width `{bits}`")
            }
            Self::SignedNonInteger(name) => {
                write!(formatter, "non-integer primitive `{name}` declares signedness")
            }
        }
    }
}

impl std::error::Error for PrimitiveError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(
        id: u64,
        name: &str,
        category: &str,
        representation: &str,
        bit_width: Option<u16>,
        default_literal: bool,
    ) -> PrimitiveMetadata {
        PrimitiveMetadata {
            id: PrimitiveId::from_declaration_id(id),
            name: name.into(),
            category: Some(category.into()),
            representation: Some(representation.into()),
            bits: bit_width.map(i64::from),
            signed: (category == "integer").then_some(true),
            default_literal: Some(default_literal),
        }
    }

    #[test]
    fn annotation_and_inference_use_the_registered_identity() {
        let catalog = PrimitiveCatalog::from_metadata([metadata(
            32,
            "i32",
            "integer",
            "fixed-integer",
            Some(32),
            true,
        )])
        .unwrap();
        assert_eq!(
            catalog.resolve("i32"),
            catalog.default_for(PrimitiveCategory::Integer)
        );
    }

    #[test]
    fn arbitrary_new_declarations_need_no_model_enum_variant() {
        let catalog = PrimitiveCatalog::from_metadata([metadata(
            128,
            "f128",
            "float",
            "ieee-float",
            Some(128),
            false,
        )])
        .unwrap();
        assert_eq!(
            catalog.definition(PrimitiveId(128)).unwrap().bit_width,
            Some(128)
        );
    }

    #[test]
    fn bytes_remains_a_distinct_registered_primitive() {
        let catalog = PrimitiveCatalog::from_metadata([metadata(
            7,
            "bytes",
            "bytes",
            "byte-string",
            None,
            false,
        )])
        .unwrap();
        let bytes = catalog
            .definition(catalog.resolve("bytes").unwrap())
            .unwrap();
        assert_eq!(bytes.category, PrimitiveCategory::Bytes);
    }

    #[test]
    fn missing_bootstrap_catalog_is_a_hard_error() {
        assert_eq!(
            PrimitiveCatalog::from_metadata([]),
            Err(PrimitiveError::EmptyCatalog)
        );
    }
}
