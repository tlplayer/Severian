use crate::{
    AbiSchemaId, AbiSignatureExpr, AbiType, AddressSpaceId, EnumType, FloatType, IntType,
    Mutability, Nullability, OpaqueId, RecordId, RecordRepr, ResourceId, SchemaParamId, UnionId,
};

/// Generic parameters supported by ABI schemas. Type, const, and address-space
/// parameters cover structures such as `View[T, Space]` and `[T; N]` without
/// adding Tensor/Data-specific concepts to the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SchemaParamKind {
    Type,
    Const,
    AddressSpace,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SchemaParam {
    pub id: SchemaParamId,
    pub name: String,
    pub kind: SchemaParamKind,
}

impl SchemaParam {
    pub fn ty(id: u16, name: impl Into<String>) -> Self {
        Self { id: SchemaParamId::new(id), name: name.into(), kind: SchemaParamKind::Type }
    }

    pub fn constant(id: u16, name: impl Into<String>) -> Self {
        Self { id: SchemaParamId::new(id), name: name.into(), kind: SchemaParamKind::Const }
    }

    pub fn address_space(id: u16, name: impl Into<String>) -> Self {
        Self { id: SchemaParamId::new(id), name: name.into(), kind: SchemaParamKind::AddressSpace }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiArgument {
    Type(AbiType),
    Const(u64),
    AddressSpace(AddressSpaceId),
}

impl AbiArgument {
    pub fn kind(&self) -> SchemaParamKind {
        match self {
            Self::Type(_) => SchemaParamKind::Type,
            Self::Const(_) => SchemaParamKind::Const,
            Self::AddressSpace(_) => SchemaParamKind::AddressSpace,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiArgumentExpr {
    Type(AbiTypeExpr),
    Const(AbiConstExpr),
    AddressSpace(AbiAddressSpaceExpr),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiConstExpr {
    Value(u64),
    Param(SchemaParamId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiAddressSpaceExpr {
    Value(AddressSpaceId),
    Param(SchemaParamId),
}

impl AbiAddressSpaceExpr {
    pub fn default_space() -> Self {
        Self::Value(AddressSpaceId::default_space())
    }
}

/// Generic/uninstantiated ABI type expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiTypeExpr {
    Unit,
    Int(IntType),
    Float(FloatType),
    TypeParam(SchemaParamId),
    Pointer(PointerTypeExpr),
    Array(ArrayTypeExpr),
    Record(RecordTypeExpr),
    Union(UnionTypeExpr),
    Enum(EnumType),
    Function(Box<AbiSignatureExpr>),
    Resource(ResourceTypeExpr),
    Opaque(OpaqueId),
    Apply(SchemaApplication),
}

impl AbiTypeExpr {
    pub fn concrete(ty: AbiType) -> Self {
        match ty {
            AbiType::Unit => Self::Unit,
            AbiType::Int(v) => Self::Int(v),
            AbiType::Float(v) => Self::Float(v),
            AbiType::Pointer(v) => Self::Pointer(PointerTypeExpr {
                pointee: Box::new(Self::concrete(*v.pointee)),
                mutability: v.mutability,
                nullability: v.nullability,
                address_space: AbiAddressSpaceExpr::Value(v.address_space),
            }),
            AbiType::Array(v) => Self::Array(ArrayTypeExpr {
                element: Box::new(Self::concrete(*v.element)),
                length: AbiConstExpr::Value(v.length),
            }),
            AbiType::Record(v) => Self::Record(RecordTypeExpr {
                id: v.id,
                repr: v.repr,
                fields: v.fields.into_iter().map(|f| RecordFieldExpr {
                    name: f.name,
                    ty: Self::concrete(f.ty),
                }).collect(),
            }),
            AbiType::Union(v) => Self::Union(UnionTypeExpr {
                id: v.id,
                fields: v.fields.into_iter().map(|f| UnionFieldExpr {
                    name: f.name,
                    ty: Self::concrete(f.ty),
                }).collect(),
            }),
            AbiType::Enum(v) => Self::Enum(v),
            AbiType::Function(v) => Self::Function(Box::new(AbiSignatureExpr {
                abi: v.abi,
                parameters: v.parameters.into_iter().map(|p| crate::AbiParameterExpr {
                    name: p.name,
                    mode: p.mode,
                    value: crate::AbiValueExpr {
                        ty: Self::concrete(p.value.ty),
                        ownership: p.value.ownership,
                        lifetime: p.value.lifetime,
                    },
                }).collect(),
                returns: crate::AbiValueExpr {
                    ty: Self::concrete(v.returns.ty),
                    ownership: v.returns.ownership,
                    lifetime: v.returns.lifetime,
                },
                variadic: v.variadic,
            })),
            AbiType::Resource(v) => Self::Resource(ResourceTypeExpr {
                id: v.id,
                repr: match v.repr {
                    crate::ResourceRepr::Pointer { address_space } => ResourceReprExpr::Pointer {
                        address_space: AbiAddressSpaceExpr::Value(address_space),
                    },
                    crate::ResourceRepr::Integer(int) => ResourceReprExpr::Integer(int),
                },
            }),
            AbiType::Opaque(v) => Self::Opaque(v),
        }
    }

    pub fn pointer_to(pointee: AbiTypeExpr) -> Self {
        Self::Pointer(PointerTypeExpr::new(pointee))
    }

    pub fn apply(schema: AbiSchemaId, arguments: Vec<AbiArgumentExpr>) -> Self {
        Self::Apply(SchemaApplication { schema, arguments })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerTypeExpr {
    pub pointee: Box<AbiTypeExpr>,
    pub mutability: Mutability,
    pub nullability: Nullability,
    pub address_space: AbiAddressSpaceExpr,
}

impl PointerTypeExpr {
    pub fn new(pointee: AbiTypeExpr) -> Self {
        Self {
            pointee: Box::new(pointee),
            mutability: Mutability::Const,
            nullability: Nullability::NonNull,
            address_space: AbiAddressSpaceExpr::default_space(),
        }
    }

    pub fn mutable(mut self) -> Self {
        self.mutability = Mutability::Mutable;
        self
    }

    pub fn nullable(mut self) -> Self {
        self.nullability = Nullability::Nullable;
        self
    }

    pub fn in_address_space(mut self, address_space: AbiAddressSpaceExpr) -> Self {
        self.address_space = address_space;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayTypeExpr {
    pub element: Box<AbiTypeExpr>,
    pub length: AbiConstExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RecordTypeExpr {
    pub id: RecordId,
    pub repr: RecordRepr,
    pub fields: Vec<RecordFieldExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RecordFieldExpr {
    pub name: String,
    pub ty: AbiTypeExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnionTypeExpr {
    pub id: UnionId,
    pub fields: Vec<UnionFieldExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnionFieldExpr {
    pub name: String,
    pub ty: AbiTypeExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceReprExpr {
    Pointer { address_space: AbiAddressSpaceExpr },
    Integer(IntType),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceTypeExpr {
    pub id: ResourceId,
    pub repr: ResourceReprExpr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SchemaApplication {
    pub schema: AbiSchemaId,
    pub arguments: Vec<AbiArgumentExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiSchema {
    pub id: AbiSchemaId,
    pub parameters: Vec<SchemaParam>,
    pub body: AbiTypeExpr,
}

impl AbiSchema {
    pub fn new(id: AbiSchemaId, parameters: Vec<SchemaParam>, body: AbiTypeExpr) -> Self {
        Self { id, parameters, body }
    }
}

/// Keeps the logical identity and arguments of a root schema application while
/// exposing the fully-expanded concrete ABI type used by layout/lowering.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiInstance {
    pub schema: AbiSchemaId,
    pub arguments: Vec<AbiArgument>,
    pub ty: AbiType,
}
