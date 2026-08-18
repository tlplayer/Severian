use crate::{AddressSpaceId, AbiSignature, OpaqueId, RecordId, ResourceId, UnionId};

/// A fully-instantiated ABI type. No generic parameters can occur here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiType {
    Unit,
    Int(IntType),
    Float(FloatType),
    Pointer(PointerType),
    Array(ArrayType),
    Record(RecordType),
    Union(UnionType),
    Enum(EnumType),
    Function(Box<AbiSignature>),
    Resource(ResourceType),
    Opaque(OpaqueId),
}

impl AbiType {
    pub const fn i8() -> Self { Self::Int(IntType::signed(8)) }
    pub const fn u8() -> Self { Self::Int(IntType::unsigned(8)) }
    pub const fn i16() -> Self { Self::Int(IntType::signed(16)) }
    pub const fn u16() -> Self { Self::Int(IntType::unsigned(16)) }
    pub const fn i32() -> Self { Self::Int(IntType::signed(32)) }
    pub const fn u32() -> Self { Self::Int(IntType::unsigned(32)) }
    pub const fn i64() -> Self { Self::Int(IntType::signed(64)) }
    pub const fn u64() -> Self { Self::Int(IntType::unsigned(64)) }
    pub const fn isize() -> Self { Self::Int(IntType::signed_pointer()) }
    pub const fn usize() -> Self { Self::Int(IntType::unsigned_pointer()) }
    pub const fn f16() -> Self { Self::Float(FloatType::F16) }
    pub const fn bf16() -> Self { Self::Float(FloatType::BF16) }
    pub const fn f32() -> Self { Self::Float(FloatType::F32) }
    pub const fn f64() -> Self { Self::Float(FloatType::F64) }

    pub fn pointer_to(pointee: AbiType) -> Self {
        Self::Pointer(PointerType::new(pointee))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntWidth {
    Fixed(u16),
    Pointer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntType {
    pub signed: bool,
    pub width: IntWidth,
}

impl IntType {
    pub const fn signed(bits: u16) -> Self {
        Self { signed: true, width: IntWidth::Fixed(bits) }
    }

    pub const fn unsigned(bits: u16) -> Self {
        Self { signed: false, width: IntWidth::Fixed(bits) }
    }

    pub const fn signed_pointer() -> Self {
        Self { signed: true, width: IntWidth::Pointer }
    }

    pub const fn unsigned_pointer() -> Self {
        Self { signed: false, width: IntWidth::Pointer }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatType {
    F16,
    BF16,
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mutability {
    Const,
    Mutable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Nullability {
    NonNull,
    Nullable,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub pointee: Box<AbiType>,
    pub mutability: Mutability,
    pub nullability: Nullability,
    pub address_space: AddressSpaceId,
}

impl PointerType {
    pub fn new(pointee: AbiType) -> Self {
        Self {
            pointee: Box::new(pointee),
            mutability: Mutability::Const,
            nullability: Nullability::NonNull,
            address_space: AddressSpaceId::default_space(),
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

    pub fn in_address_space(mut self, address_space: AddressSpaceId) -> Self {
        self.address_space = address_space;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub element: Box<AbiType>,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RecordRepr {
    C,
    Packed,
    Transparent,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RecordType {
    pub id: RecordId,
    pub repr: RecordRepr,
    pub fields: Vec<RecordField>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RecordField {
    pub name: String,
    pub ty: AbiType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnionType {
    pub id: UnionId,
    pub fields: Vec<UnionField>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnionField {
    pub name: String,
    pub ty: AbiType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumType {
    pub repr: IntType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceRepr {
    Pointer { address_space: AddressSpaceId },
    Integer(IntType),
}

impl ResourceRepr {
    pub fn pointer() -> Self {
        Self::Pointer { address_space: AddressSpaceId::default_space() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceType {
    pub id: ResourceId,
    pub repr: ResourceRepr,
}
