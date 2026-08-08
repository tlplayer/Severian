//! Semantic capability/trait classification.
//!
//! These are compiler traits, not user-defined language traits yet. They let
//! optimization and lowering ask stable questions such as "is this type
//! numeric?", "can it be indexed?", or "can this value use the StableHLO
//! path?" without repeating large `match ValueType` blocks.

use severian_hir::{TensorElementType, TensorType, ValueType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SemanticTrait {
    Numeric = 0,
    Integer = 1,
    Floating = 2,
    Boolean = 3,
    Ordered = 4,
    Equatable = 5,
    Iterable = 6,
    Collection = 7,
    Indexable = 8,
    Sliceable = 9,
    Callable = 10,
    Tensor = 11,
    Channel = 12,
    Cloneable = 13,
    Movable = 14,
    Borrowable = 15,
    Sendable = 16,
    Shareable = 17,
    SimdCompatible = 18,
    GpuCompatible = 19,
    StableHloCompatible = 20,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TraitSet(u64);

impl TraitSet {
    pub const EMPTY: Self = Self(0);

    pub const fn singleton(trait_: SemanticTrait) -> Self {
        Self(1u64 << trait_ as u8)
    }

    pub const fn contains(self, trait_: SemanticTrait) -> bool {
        (self.0 & (1u64 << trait_ as u8)) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn insert(&mut self, trait_: SemanticTrait) {
        self.0 |= 1u64 << trait_ as u8;
    }

    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::ops::BitOr for TraitSet {
    type Output = TraitSet;

    fn bitor(self, rhs: TraitSet) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for TraitSet {
    fn bitor_assign(&mut self, rhs: TraitSet) {
        self.0 |= rhs.0;
    }
}

pub fn traits_for_type(ty: ValueType) -> TraitSet {
    let mut traits = TraitSet::EMPTY;

    // All first-class Severian values participate in ownership operations.
    traits.insert(SemanticTrait::Cloneable);
    traits.insert(SemanticTrait::Movable);
    traits.insert(SemanticTrait::Borrowable);

    match ty {
        ValueType::Int => {
            traits.insert(SemanticTrait::Numeric);
            traits.insert(SemanticTrait::Integer);
            traits.insert(SemanticTrait::Ordered);
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
            traits.insert(SemanticTrait::SimdCompatible);
            traits.insert(SemanticTrait::GpuCompatible);
            traits.insert(SemanticTrait::StableHloCompatible);
        }

        ValueType::Float => {
            traits.insert(SemanticTrait::Numeric);
            traits.insert(SemanticTrait::Floating);
            traits.insert(SemanticTrait::Ordered);
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
            traits.insert(SemanticTrait::SimdCompatible);
            traits.insert(SemanticTrait::GpuCompatible);
            traits.insert(SemanticTrait::StableHloCompatible);
        }

        ValueType::Bool => {
            traits.insert(SemanticTrait::Boolean);
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
            traits.insert(SemanticTrait::SimdCompatible);
            traits.insert(SemanticTrait::GpuCompatible);
            traits.insert(SemanticTrait::StableHloCompatible);
        }

        ValueType::String => {
            traits.insert(SemanticTrait::Ordered);
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Iterable);
            traits.insert(SemanticTrait::Collection);
            traits.insert(SemanticTrait::Indexable);
            traits.insert(SemanticTrait::Sliceable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
        }

        ValueType::List | ValueType::Tuple => {
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Iterable);
            traits.insert(SemanticTrait::Collection);
            traits.insert(SemanticTrait::Indexable);
            traits.insert(SemanticTrait::Sliceable);
            traits.insert(SemanticTrait::Sendable);
        }

        ValueType::Map => {
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Iterable);
            traits.insert(SemanticTrait::Collection);
            traits.insert(SemanticTrait::Indexable);
            traits.insert(SemanticTrait::Sendable);
        }

        ValueType::Set => {
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Iterable);
            traits.insert(SemanticTrait::Collection);
            traits.insert(SemanticTrait::Sendable);
        }

        ValueType::Tensor(tensor) => {
            traits |= traits_for_tensor(tensor);
        }

        ValueType::Channel => {
            traits.insert(SemanticTrait::Channel);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
        }

        ValueType::Function => {
            traits.insert(SemanticTrait::Callable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
        }

        ValueType::Result | ValueType::Option => {
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Sendable);
        }

        ValueType::Any => {
            // `Any` is intentionally conservative: ownership operations are
            // known, representation-specific traits are not.
        }

        ValueType::Unit => {
            traits.insert(SemanticTrait::Equatable);
            traits.insert(SemanticTrait::Sendable);
            traits.insert(SemanticTrait::Shareable);
        }
    }

    traits
}

pub fn traits_for_tensor(tensor: TensorType) -> TraitSet {
    let mut traits = TraitSet::EMPTY;

    traits.insert(SemanticTrait::Tensor);
    traits.insert(SemanticTrait::Iterable);
    traits.insert(SemanticTrait::Collection);
    traits.insert(SemanticTrait::Indexable);
    traits.insert(SemanticTrait::Sliceable);
    traits.insert(SemanticTrait::Cloneable);
    traits.insert(SemanticTrait::Movable);
    traits.insert(SemanticTrait::Borrowable);
    traits.insert(SemanticTrait::Sendable);
    traits.insert(SemanticTrait::GpuCompatible);
    traits.insert(SemanticTrait::StableHloCompatible);

    match tensor.element {
        TensorElementType::F32 | TensorElementType::F64 => {
            traits.insert(SemanticTrait::Numeric);
            traits.insert(SemanticTrait::Floating);
            traits.insert(SemanticTrait::SimdCompatible);
        }
        TensorElementType::I32 | TensorElementType::I64 => {
            traits.insert(SemanticTrait::Numeric);
            traits.insert(SemanticTrait::Integer);
            traits.insert(SemanticTrait::SimdCompatible);
        }
    }

    traits
}

pub fn implements(ty: ValueType, trait_: SemanticTrait) -> bool {
    traits_for_type(ty).contains(trait_)
}

pub fn is_numeric(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::Numeric)
}

pub fn is_collection(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::Collection)
}

pub fn is_gpu_compatible(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::GpuCompatible)
}

pub fn is_stablehlo_compatible(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::StableHloCompatible)
}

pub fn can_binary_arithmetic(left: ValueType, right: ValueType) -> bool {
    match (left, right) {
        (ValueType::Tensor(left), ValueType::Tensor(right)) => {
            left.broadcast_with(right).is_ok()
        }

        (ValueType::Tensor(tensor), scalar)
        | (scalar, ValueType::Tensor(tensor)) => {
            scalar_matches_tensor_element(scalar, tensor.element)
        }

        _ => is_numeric(left) && is_numeric(right),
    }
}

pub fn can_compare(left: ValueType, right: ValueType) -> bool {
    if left == right {
        return implements(left, SemanticTrait::Equatable);
    }

    matches!(
        (left, right),
        (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int)
    )
}

pub fn can_order(left: ValueType, right: ValueType) -> bool {
    if left == right {
        return implements(left, SemanticTrait::Ordered);
    }

    matches!(
        (left, right),
        (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int)
    )
}

pub fn can_index(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::Indexable)
}

pub fn can_slice(ty: ValueType) -> bool {
    implements(ty, SemanticTrait::Sliceable)
}

fn scalar_matches_tensor_element(
    scalar: ValueType,
    element: TensorElementType,
) -> bool {
    matches!(
        (scalar, element),
        (ValueType::Float, TensorElementType::F32 | TensorElementType::F64)
            | (ValueType::Int, TensorElementType::I32 | TensorElementType::I64)
    )
}
