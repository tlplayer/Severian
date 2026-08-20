use crate::{AnyOrigin, TensorType, ValueType};
use std::{collections::BTreeMap, path::PathBuf};

macro_rules! stable_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);

        impl $name {
            pub fn from_name(name: &str) -> Self {
                Self(stable_name_hash(name))
            }

            pub fn in_namespace(self, namespace: &str) -> Self {
                Self::from_name(&format!("{namespace}#{}", self.0))
            }
        }
    };
}

stable_id!(FunctionId);
stable_id!(TypeDefinitionId);
stable_id!(VariantId);
stable_id!(BindingId);
stable_id!(FieldId);
stable_id!(MethodId);
stable_id!(ModuleId);
stable_id!(PackageId);
stable_id!(SymbolId);
stable_id!(IntrinsicId);

/// Resolved identity for a local declaration and all of its uses.
///
/// `name` exists for diagnostics and emitted debug names. Compiler state after
/// resolution must key correctness decisions by `id`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingRef {
    pub id: BindingId,
    pub name: String,
}

impl BindingRef {
    pub fn new(id: BindingId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }

    pub fn source(name: impl Into<String>, start: usize, end: usize) -> Self {
        let name = name.into();
        let id = BindingId::from_name(&format!("{name}@{start}:{end}"));
        Self { id, name }
    }

    pub fn synthetic(name: impl Into<String>) -> Self {
        let name = name.into();
        let id = BindingId::from_name(&name);
        Self { id, name }
    }
}

impl From<String> for BindingRef {
    fn from(name: String) -> Self {
        Self::synthetic(name)
    }
}

impl From<&str> for BindingRef {
    fn from(name: &str) -> Self {
        Self::synthetic(name)
    }
}

impl std::fmt::Display for BindingRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(formatter)
    }
}

impl AsRef<str> for BindingRef {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u64);

impl HirId {
    pub const fn from_source_range(start: usize, end: usize) -> Self {
        Self(((start as u64) << 32) ^ end as u64)
    }

    pub const fn synthetic(value: u64) -> Self {
        Self(u64::MAX - value)
    }

    pub fn from_source_span(file: SourceFileId, range: SourceRange) -> Self {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in file
            .0
            .to_le_bytes()
            .into_iter()
            .chain(range.start.to_le_bytes())
            .chain(range.end.to_le_bytes())
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Self(hash)
    }

    pub const fn legacy_source_range(self) -> Option<SourceRange> {
        if self.0 > u64::MAX - (1 << 20) {
            return None;
        }
        Some(SourceRange {
            start: (self.0 >> 32) as usize,
            end: (self.0 & u32::MAX as u64) as usize,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceFileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceSpan {
    pub file: SourceFileId,
    pub range: SourceRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub id: SourceFileId,
    pub path: PathBuf,
    pub source: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceMap {
    files: Vec<SourceFile>,
    pub(crate) expression_spans: BTreeMap<HirId, SourceSpan>,
    definition_spans: BTreeMap<DefinitionId, SourceSpan>,
}

impl SourceMap {
    pub fn add_file(
        &mut self,
        path: impl Into<PathBuf>,
        source: impl Into<String>,
    ) -> SourceFileId {
        let path = path.into();
        if let Some(file) = self.files.iter().find(|file| file.path == path) {
            return file.id;
        }
        let id = SourceFileId(self.files.len() as u32);
        self.files.push(SourceFile {
            id,
            path,
            source: source.into(),
        });
        id
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn file(&self, id: SourceFileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn expression_span(&self, id: HirId) -> Option<SourceSpan> {
        self.expression_spans.get(&id).copied()
    }

    pub fn definition_span(&self, id: DefinitionId) -> Option<SourceSpan> {
        self.definition_spans.get(&id).copied()
    }

    pub fn record_definition(&mut self, id: DefinitionId, span: SourceSpan) {
        self.definition_spans.insert(id, span);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefinitionId {
    Function(FunctionId),
    Type(TypeDefinitionId),
    Variant {
        owner: TypeDefinitionId,
        variant: VariantId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Primitive(PrimitiveId),
    Int,
    Float,
    Bool,
    String,
    Unit,
    Any,
    List(TypeId),
    Tuple(Vec<TypeId>),
    Map {
        key: TypeId,
        value: TypeId,
    },
    Set(TypeId),
    Tensor(TensorType),
    TensorAny,
    Channel(TypeId),
    Function {
        parameters: Vec<TypeId>,
        returns: TypeId,
    },
    Result {
        ok: TypeId,
        error: TypeId,
    },
    Option(TypeId),
    Union(Vec<TypeId>),
    Future(TypeId),
    Reference {
        mutable: bool,
        inner: TypeId,
    },
    Named {
        definition: TypeDefinitionId,
        name: String,
        arguments: Vec<TypeId>,
    },
    Unresolved {
        name: String,
        arguments: Vec<TypeId>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeTable {
    types: Vec<TypeKind>,
}

impl TypeTable {
    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if let Some(index) = self.types.iter().position(|existing| existing == &kind) {
            return TypeId(index as u32);
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(kind);
        id
    }

    pub fn get(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id.0 as usize)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &TypeKind)> {
        self.types
            .iter()
            .enumerate()
            .map(|(index, kind)| (TypeId(index as u32), kind))
    }

    pub fn legacy(&mut self, ty: ValueType) -> TypeId {
        let kind = match ty {
            ValueType::Int => TypeKind::Int,
            ValueType::Float => TypeKind::Float,
            ValueType::Bool => TypeKind::Bool,
            ValueType::String => TypeKind::String,
            ValueType::List => TypeKind::List(self.intern(TypeKind::Any)),
            ValueType::Tuple => TypeKind::Tuple(Vec::new()),
            ValueType::Map => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Map {
                    key: any,
                    value: any,
                }
            }
            ValueType::Set => TypeKind::Set(self.intern(TypeKind::Any)),
            ValueType::Tensor(tensor) => TypeKind::Tensor(tensor),
            ValueType::TensorAny => TypeKind::TensorAny,
            ValueType::Channel => TypeKind::Channel(self.intern(TypeKind::Any)),
            ValueType::Function => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Function {
                    parameters: Vec::new(),
                    returns: any,
                }
            }
            ValueType::Result => {
                let any = self.intern(TypeKind::Any);
                TypeKind::Result {
                    ok: any,
                    error: any,
                }
            }
            ValueType::Option => TypeKind::Option(self.intern(TypeKind::Any)),
            ValueType::Interface(definition) => TypeKind::Named {
                definition,
                name: format!("interface#{}", definition.0),
                arguments: Vec::new(),
            },
            ValueType::Any => TypeKind::Any,
            ValueType::Unit => TypeKind::Unit,
        };
        self.intern(kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailedFunctionType {
    pub parameters: Vec<TypeId>,
    pub returns: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDefinition {
    pub name: String,
    pub ty: TypeId,
    pub default: Option<HirId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDefinition {
    pub id: TypeDefinitionId,
    pub name: String,
    pub fields: Vec<FieldDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDefinition {
    pub id: VariantId,
    pub name: String,
    pub fields: Vec<TypeId>,
    pub transitions: Vec<VariantId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDefinition {
    pub id: TypeDefinitionId,
    pub name: String,
    pub variants: Vec<VariantDefinition>,
}

/// The implementation set for one semantic trait after the reachable package
/// graph has been closed. Providers and properties are stored in deterministic
/// order so builds never depend on source discovery order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitRegistryDefinition {
    pub name: String,
    pub properties: Vec<TraitPropertyDefinition>,
    pub implementations: Vec<TraitImplementationDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitPropertyDefinition {
    pub name: String,
    pub ty: String,
    pub default: Option<TraitPropertyValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitImplementationDefinition {
    pub name: String,
    pub properties: BTreeMap<String, TraitPropertyValue>,
}

/// Closed, compiler-readable constants accepted in trait registry properties.
/// `Float` retains IEEE bits so registry metadata remains equality comparable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraitPropertyValue {
    Integer(i64),
    Float(u64),
    Boolean(bool),
    String(String),
    Symbol(String),
    Constructor {
        name: String,
        arguments: Vec<TraitPropertyValue>,
    },
    List(Vec<TraitPropertyValue>),
    Set(Vec<TraitPropertyValue>),
    Tuple(Vec<TraitPropertyValue>),
    Map(Vec<(TraitPropertyValue, TraitPropertyValue)>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramMetadata {
    pub sources: SourceMap,
    pub types: TypeTable,
    /// Primitive contracts loaded before ordinary semantic analysis.
    pub primitives: BTreeMap<PrimitiveId, PrimitiveDefinition>,
    pub expression_types: BTreeMap<HirId, TypeId>,
    pub expression_any_origins: BTreeMap<HirId, AnyOrigin>,
    pub globals: BTreeMap<String, TypeId>,
    pub functions: BTreeMap<FunctionId, DetailedFunctionType>,
    pub classes: BTreeMap<TypeDefinitionId, ClassDefinition>,
    pub enums: BTreeMap<TypeDefinitionId, EnumDefinition>,
    /// Statically closed trait provider sets for this package graph.
    pub trait_registries: BTreeMap<String, TraitRegistryDefinition>,
    /// Package-owned external functions keyed by their provider symbol.
    pub external_functions: BTreeMap<String, severian_abi::ExternalFunction>,
}

fn stable_name_hash(name: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
