#![forbid(unsafe_code)]

/// Canonical scalar element types shared by source typing, model artifacts,
/// compiler IR and execution backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8E4M3FN,
    F8E5M2,
    F16,
    BF16,
    F32,
    F64,
    C64,
    C128,
}

/// Compatibility name retained while tensor HIR migrates to `DType`.
pub type TensorElementType = DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeClass {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
}

pub type TensorElementClass = DTypeClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeConstraint {
    Any,
    Numeric,
    Integer,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
}

pub type TensorElementConstraint = DTypeConstraint;

impl DType {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "bool" => Self::Bool,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" | "int" => Self::I64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "f8e4m3" | "f8e4m3fn" => Self::F8E4M3FN,
            "f8e5m2" => Self::F8E5M2,
            "f16" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::BF16,
            "f32" | "float32" => Self::F32,
            "f64" | "float" | "float64" => Self::F64,
            "c64" | "complex64" => Self::C64,
            "c128" | "complex128" => Self::C128,
            _ => return None,
        })
    }

    pub fn parse_safetensors(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "U8" => Self::U8,
            "U16" => Self::U16,
            "U32" => Self::U32,
            "U64" => Self::U64,
            "F8_E4M3" | "F8_E4M3FN" => Self::F8E4M3FN,
            "F8_E5M2" => Self::F8E5M2,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "C64" => Self::C64,
            "C128" => Self::C128,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F8E4M3FN => "f8e4m3fn",
            Self::F8E5M2 => "f8e5m2",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        }
    }

    pub const fn mlir_name(self) -> &'static str {
        match self {
            Self::Bool => "i1",
            Self::F8E4M3FN => "f8E4M3FN",
            Self::F8E5M2 => "f8E5M2",
            Self::C64 => "complex<f32>",
            Self::C128 => "complex<f64>",
            _ => self.name(),
        }
    }

    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::F8E4M3FN => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::C128 => "C128",
        }
    }

    pub const fn storage_bytes(self) -> u8 {
        match self {
            Self::Bool | Self::I8 | Self::U8 | Self::F8E4M3FN | Self::F8E5M2 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub const fn byte_width(self) -> usize {
        self.storage_bytes() as usize
    }

    pub const fn class(self) -> DTypeClass {
        match self {
            Self::Bool => DTypeClass::Boolean,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => DTypeClass::SignedInteger,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => DTypeClass::UnsignedInteger,
            Self::F8E4M3FN | Self::F8E5M2 | Self::F16 | Self::BF16 | Self::F32 | Self::F64 => {
                DTypeClass::Float
            }
            Self::C64 | Self::C128 => DTypeClass::Complex,
        }
    }

    pub const fn satisfies(self, constraint: DTypeConstraint) -> bool {
        use DTypeClass as Class;
        use DTypeConstraint as Constraint;
        matches!(constraint, Constraint::Any)
            || matches!(
                (self.class(), constraint),
                (
                    Class::SignedInteger | Class::UnsignedInteger | Class::Float | Class::Complex,
                    Constraint::Numeric
                ) | (
                    Class::SignedInteger | Class::UnsignedInteger,
                    Constraint::Integer
                ) | (Class::SignedInteger, Constraint::SignedInteger)
                    | (Class::UnsignedInteger, Constraint::UnsignedInteger)
                    | (Class::Float, Constraint::Float)
                    | (Class::Complex, Constraint::Complex)
            )
    }

    /// Severian's language-level promotion rule. Backends consume this result
    /// rather than applying implementation-specific implicit conversions.
    pub const fn promote(left: Self, right: Self) -> Option<Self> {
        use DType as T;
        if left as u8 == right as u8 {
            return Some(left);
        }
        if matches!(left, T::Bool) || matches!(right, T::Bool) {
            return None;
        }
        match (left, right) {
            (T::C128, _) | (_, T::C128) => Some(T::C128),
            (T::C64, T::F64) | (T::F64, T::C64) => Some(T::C128),
            (T::C64, _) | (_, T::C64) => Some(T::C64),
            (T::F64, _) | (_, T::F64) => Some(T::F64),
            (T::F32, _) | (_, T::F32) => Some(T::F32),
            (T::BF16, T::F16) | (T::F16, T::BF16) => Some(T::F32),
            (T::BF16, _) | (_, T::BF16) => Some(T::BF16),
            (T::F16, _) | (_, T::F16) => Some(T::F16),
            (T::F8E4M3FN, T::F8E5M2) | (T::F8E5M2, T::F8E4M3FN) => Some(T::F16),
            (T::F8E4M3FN, _) | (_, T::F8E4M3FN) => Some(T::F8E4M3FN),
            (T::F8E5M2, _) | (_, T::F8E5M2) => Some(T::F8E5M2),
            _ => Self::promote_integers(left, right),
        }
    }

    const fn integer_width(self) -> Option<(u8, bool)> {
        match self {
            Self::I8 => Some((8, true)),
            Self::I16 => Some((16, true)),
            Self::I32 => Some((32, true)),
            Self::I64 => Some((64, true)),
            Self::U8 => Some((8, false)),
            Self::U16 => Some((16, false)),
            Self::U32 => Some((32, false)),
            Self::U64 => Some((64, false)),
            _ => None,
        }
    }

    const fn promote_integers(left: Self, right: Self) -> Option<Self> {
        let (Some((left_width, left_signed)), Some((right_width, right_signed))) =
            (left.integer_width(), right.integer_width())
        else {
            return None;
        };
        let width = if left_width > right_width {
            left_width
        } else {
            right_width
        };
        if left_signed == right_signed {
            return match (width, left_signed) {
                (8, true) => Some(Self::I8),
                (16, true) => Some(Self::I16),
                (32, true) => Some(Self::I32),
                (64, true) => Some(Self::I64),
                (8, false) => Some(Self::U8),
                (16, false) => Some(Self::U16),
                (32, false) => Some(Self::U32),
                (64, false) => Some(Self::U64),
                _ => None,
            };
        }
        let signed_width = if left_signed { left_width } else { right_width };
        let unsigned_width = if left_signed { right_width } else { left_width };
        let required = if signed_width > unsigned_width {
            signed_width
        } else if unsigned_width < 64 {
            unsigned_width * 2
        } else {
            return None;
        };
        match required {
            8 => Some(Self::I8),
            16 => Some(Self::I16),
            32 => Some(Self::I32),
            64 => Some(Self::I64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_external_and_language_spellings() {
        assert_eq!(DType::parse("bf16"), Some(DType::BF16));
        assert_eq!(DType::parse_safetensors("BF16"), Some(DType::BF16));
        assert_eq!(DType::parse_safetensors("F8_E4M3FN"), Some(DType::F8E4M3FN));
    }

    #[test]
    fn promotion_is_centralized() {
        assert_eq!(DType::promote(DType::F16, DType::BF16), Some(DType::F32));
        assert_eq!(DType::promote(DType::I8, DType::U8), Some(DType::I16));
        assert_eq!(DType::promote(DType::I64, DType::U64), None);
    }
}
