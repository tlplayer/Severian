use crate::{
    BinaryOperator, LiteralKind, OperatorSignature, PrimitiveId, TypeContextBuilder, TypeError,
    TypeId, TypePattern, UnaryOperator,
};

const PATH: &str = "universal.primitive";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveCategory {
    Boolean,
    Character,
    Integer,
    Float,
    Measured,
    Text,
    Bytes,
    Absence,
    Unit,
    Arguments,
}

impl PrimitiveCategory {
    pub const fn literal_kind(self) -> Option<LiteralKind> {
        match self {
            Self::Boolean => Some(LiteralKind::Boolean),
            Self::Character => Some(LiteralKind::Character),
            Self::Integer => Some(LiteralKind::Integer),
            Self::Float => Some(LiteralKind::Float),
            Self::Measured | Self::Arguments => None,
            Self::Text => Some(LiteralKind::String),
            Self::Bytes => Some(LiteralKind::Bytes),
            Self::Absence => Some(LiteralKind::None),
            Self::Unit => Some(LiteralKind::Unit),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerWidth {
    Fixed(u16),
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatFormat {
    Ieee(u16),
    BrainFloat16,
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveRepresentation {
    Integer { bits: IntegerWidth, signed: bool },
    Float { format: FloatFormat },
    PointerInteger { signed: bool },
    Boolean,
    Character,
    String,
    Bytes,
    None,
    Unit,
    Arguments,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveDefinition {
    pub id: PrimitiveId,
    pub type_id: TypeId,
    pub category: PrimitiveCategory,
    pub representation: PrimitiveRepresentation,
    pub default_literal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveSpec {
    pub name: &'static str,
    pub category: PrimitiveCategory,
    pub representation: PrimitiveRepresentation,
    pub default_literal: bool,
}

const fn fixed_integer(bits: u16, signed: bool) -> PrimitiveRepresentation {
    PrimitiveRepresentation::Integer {
        bits: IntegerWidth::Fixed(bits),
        signed,
    }
}

const fn ieee_float(bits: u16) -> PrimitiveRepresentation {
    PrimitiveRepresentation::Float {
        format: FloatFormat::Ieee(bits),
    }
}

pub const PRIMITIVES: &[PrimitiveSpec] = &[
    PrimitiveSpec {
        name: "bool",
        category: PrimitiveCategory::Boolean,
        representation: PrimitiveRepresentation::Boolean,
        default_literal: true,
    },
    PrimitiveSpec {
        name: "char",
        category: PrimitiveCategory::Character,
        representation: PrimitiveRepresentation::Character,
        default_literal: true,
    },
    PrimitiveSpec {
        name: "int",
        category: PrimitiveCategory::Integer,
        representation: PrimitiveRepresentation::Integer {
            bits: IntegerWidth::Machine,
            signed: true,
        },
        default_literal: true,
    },
    PrimitiveSpec {
        name: "i8",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(8, true),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "i16",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(16, true),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "i32",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(32, true),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "i64",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(64, true),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "i128",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(128, true),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "isize",
        category: PrimitiveCategory::Integer,
        representation: PrimitiveRepresentation::PointerInteger { signed: true },
        default_literal: false,
    },
    PrimitiveSpec {
        name: "u8",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(8, false),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "u16",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(16, false),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "u32",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(32, false),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "u64",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(64, false),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "u128",
        category: PrimitiveCategory::Integer,
        representation: fixed_integer(128, false),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "usize",
        category: PrimitiveCategory::Integer,
        representation: PrimitiveRepresentation::PointerInteger { signed: false },
        default_literal: false,
    },
    PrimitiveSpec {
        name: "float",
        category: PrimitiveCategory::Float,
        representation: PrimitiveRepresentation::Float {
            format: FloatFormat::Machine,
        },
        default_literal: true,
    },
    PrimitiveSpec {
        name: "f16",
        category: PrimitiveCategory::Float,
        representation: ieee_float(16),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "bf16",
        category: PrimitiveCategory::Float,
        representation: PrimitiveRepresentation::Float {
            format: FloatFormat::BrainFloat16,
        },
        default_literal: false,
    },
    PrimitiveSpec {
        name: "f32",
        category: PrimitiveCategory::Float,
        representation: ieee_float(32),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "f64",
        category: PrimitiveCategory::Float,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "string",
        category: PrimitiveCategory::Text,
        representation: PrimitiveRepresentation::String,
        default_literal: true,
    },
    PrimitiveSpec {
        name: "Error",
        category: PrimitiveCategory::Text,
        representation: PrimitiveRepresentation::String,
        default_literal: false,
    },
    PrimitiveSpec {
        name: "bytes",
        category: PrimitiveCategory::Bytes,
        representation: PrimitiveRepresentation::Bytes,
        default_literal: false,
    },
    PrimitiveSpec {
        name: "None",
        category: PrimitiveCategory::Absence,
        representation: PrimitiveRepresentation::None,
        default_literal: true,
    },
    PrimitiveSpec {
        name: "unit",
        category: PrimitiveCategory::Unit,
        representation: PrimitiveRepresentation::Unit,
        default_literal: true,
    },
    PrimitiveSpec {
        name: "args",
        category: PrimitiveCategory::Arguments,
        representation: PrimitiveRepresentation::Arguments,
        default_literal: false,
    },
    PrimitiveSpec {
        name: "data_size",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "duration",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "data_rate",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "frequency",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "percentage",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "temperature",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "voltage",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "current",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
    PrimitiveSpec {
        name: "power",
        category: PrimitiveCategory::Measured,
        representation: ieee_float(64),
        default_literal: false,
    },
];

pub fn install_primitives(types: &mut TypeContextBuilder) -> Result<(), TypeError> {
    for (name, parameters) in [
        ("Primitive", 0),
        ("Integer", 1),
        ("Floating", 1),
        ("Measured", 1),
    ] {
        types.register_generic_declaration(format!("{PATH}.{name}"), name, parameters)?;
    }
    for primitive in PRIMITIVES {
        let ty =
            types.register_declaration(format!("{PATH}.{}", primitive.name), primitive.name)?;
        types.define_primitive(
            ty,
            primitive.category,
            primitive.representation,
            primitive.default_literal,
        )?;
    }

    let primitive = required(types, "Primitive")?;
    let integer = required(types, "Integer")?;
    let floating = required(types, "Floating")?;
    let measured = required(types, "Measured")?;
    for spec in PRIMITIVES {
        let ty = required(types, spec.name)?;
        types.add_capability(ty, primitive)?;
        match spec.category {
            PrimitiveCategory::Integer => types.add_capability(ty, integer)?,
            PrimitiveCategory::Float => types.add_capability(ty, floating)?,
            PrimitiveCategory::Measured => types.add_capability(ty, measured)?,
            _ => {}
        }
    }

    install_trait_operators(types, integer, floating, measured);
    install_primitive_operators(types)?;
    Ok(())
}

fn install_trait_operators(
    types: &mut TypeContextBuilder,
    integer: TypeId,
    floating: TypeId,
    measured: TypeId,
) {
    for operator in [UnaryOperator::Positive, UnaryOperator::Negative] {
        for protocol in [integer, floating, measured] {
            types.add_trait_unary(protocol, operator);
        }
    }
    for operator in [
        BinaryOperator::Add,
        BinaryOperator::Subtract,
        BinaryOperator::Equal,
        BinaryOperator::NotEqual,
        BinaryOperator::Less,
        BinaryOperator::LessEqual,
        BinaryOperator::Greater,
        BinaryOperator::GreaterEqual,
    ] {
        for protocol in [integer, floating, measured] {
            types.add_trait_binary(protocol, operator);
        }
    }
    for operator in [
        BinaryOperator::Multiply,
        BinaryOperator::Divide,
        BinaryOperator::Remainder,
        BinaryOperator::Power,
    ] {
        for protocol in [integer, floating] {
            types.add_trait_binary(protocol, operator);
        }
    }
    for operator in [
        BinaryOperator::BitwiseOr,
        BinaryOperator::BitwiseAnd,
        BinaryOperator::BitwiseXor,
    ] {
        types.add_trait_binary(integer, operator);
    }
}

fn install_primitive_operators(types: &mut TypeContextBuilder) -> Result<(), TypeError> {
    let boolean = required(types, "bool")?;
    for operator in [BinaryOperator::And, BinaryOperator::Or] {
        add_binary(types, operator, boolean, boolean, boolean);
    }
    types.add_unary(UnaryOperator::Not, boolean, boolean);
    add_comparisons(types, boolean, boolean, false);

    let character = required(types, "char")?;
    add_comparisons(types, character, boolean, true);
    let bytes = required(types, "bytes")?;
    add_comparisons(types, bytes, boolean, false);
    add_binary(types, BinaryOperator::Add, bytes, bytes, bytes);
    let string = required(types, "string")?;
    add_comparisons(types, string, boolean, true);
    add_binary(types, BinaryOperator::Add, string, string, string);
    for name in ["None", "unit"] {
        add_comparisons(types, required(types, name)?, boolean, false);
    }

    for spec in PRIMITIVES {
        let ty = required(types, spec.name)?;
        match spec.category {
            PrimitiveCategory::Integer => add_numeric(types, ty, boolean, true),
            PrimitiveCategory::Float => add_numeric(types, ty, boolean, false),
            PrimitiveCategory::Measured => add_measured(types, ty, boolean),
            _ => {}
        }
    }
    let float = required(types, "float")?;
    let duration = required(types, "duration")?;
    let frequency = required(types, "frequency")?;
    add_binary(types, BinaryOperator::Divide, float, duration, frequency);
    let data_size = required(types, "data_size")?;
    let data_rate = required(types, "data_rate")?;
    add_binary(types, BinaryOperator::Divide, data_size, data_size, float);
    add_binary(
        types,
        BinaryOperator::Divide,
        data_size,
        duration,
        data_rate,
    );
    add_binary(types, BinaryOperator::Divide, duration, duration, float);
    let percentage = required(types, "percentage")?;
    for operator in [BinaryOperator::Equal, BinaryOperator::NotEqual] {
        add_binary(types, operator, percentage, float, boolean);
    }
    Ok(())
}

fn add_numeric(types: &mut TypeContextBuilder, ty: TypeId, boolean: TypeId, bitwise: bool) {
    types.add_unary(UnaryOperator::Positive, ty, ty);
    types.add_unary(UnaryOperator::Negative, ty, ty);
    for operator in [
        BinaryOperator::Add,
        BinaryOperator::Subtract,
        BinaryOperator::Multiply,
        BinaryOperator::Divide,
        BinaryOperator::Remainder,
        BinaryOperator::Power,
    ] {
        add_binary(types, operator, ty, ty, ty);
    }
    if bitwise {
        for operator in [
            BinaryOperator::BitwiseOr,
            BinaryOperator::BitwiseAnd,
            BinaryOperator::BitwiseXor,
        ] {
            add_binary(types, operator, ty, ty, ty);
        }
    }
    add_comparisons(types, ty, boolean, true);
}

fn add_measured(types: &mut TypeContextBuilder, ty: TypeId, boolean: TypeId) {
    types.add_unary(UnaryOperator::Positive, ty, ty);
    types.add_unary(UnaryOperator::Negative, ty, ty);
    add_binary(types, BinaryOperator::Add, ty, ty, ty);
    add_binary(types, BinaryOperator::Subtract, ty, ty, ty);
    add_comparisons(types, ty, boolean, true);
}

fn add_comparisons(types: &mut TypeContextBuilder, ty: TypeId, boolean: TypeId, ordered: bool) {
    for operator in [BinaryOperator::Equal, BinaryOperator::NotEqual] {
        add_binary(types, operator, ty, ty, boolean);
    }
    if ordered {
        for operator in [
            BinaryOperator::Less,
            BinaryOperator::LessEqual,
            BinaryOperator::Greater,
            BinaryOperator::GreaterEqual,
        ] {
            add_binary(types, operator, ty, ty, boolean);
        }
    }
}

fn add_binary(
    types: &mut TypeContextBuilder,
    operator: BinaryOperator,
    left: TypeId,
    right: TypeId,
    result: TypeId,
) {
    types.add_binary(OperatorSignature {
        operator,
        left: TypePattern::Exact(left),
        right: TypePattern::Exact(right),
        result: TypePattern::Exact(result),
    });
}

fn required(types: &TypeContextBuilder, name: &str) -> Result<TypeId, TypeError> {
    types
        .resolve_name(name)
        .ok_or_else(|| TypeError::UnknownName(name.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiteralValue, TypeConstraint};

    #[test]
    fn catalog_installs_metadata_literals_capabilities_and_operators() {
        let mut builder = TypeContextBuilder::new();
        install_primitives(&mut builder).unwrap();
        let types = builder.build();
        assert_eq!(
            types
                .primitive(types.resolve_name("i32").unwrap())
                .unwrap()
                .representation,
            fixed_integer(32, true)
        );
        assert_eq!(
            types.resolve_literal(&LiteralValue::Integer("1".into()), None),
            Ok(types.resolve_name("int").unwrap())
        );
        let i32 = types.resolve_name("i32").unwrap();
        assert!(types.supports_binary(BinaryOperator::Add, i32));
        assert!(types.implements(i32, types.resolve_name("Integer").unwrap()));
    }

    #[test]
    fn primitive_identity_is_stable_and_compiler_owned() {
        let mut first = TypeContextBuilder::new();
        install_primitives(&mut first).unwrap();
        let mut second = TypeContextBuilder::new();
        install_primitives(&mut second).unwrap();
        let first = first.build();
        let second = second.build();
        let id = |types: &crate::TypeContext| {
            types
                .primitive(types.resolve_name("i32").unwrap())
                .unwrap()
                .id
        };
        assert_eq!(id(&first), id(&second));
        assert_eq!(
            first
                .definition(first.resolve_name("i32").unwrap())
                .unwrap()
                .path,
            "universal.primitive.i32"
        );
    }

    #[test]
    fn literal_operands_resolve_against_exact_primitive_signatures() {
        let mut builder = TypeContextBuilder::new();
        install_primitives(&mut builder).unwrap();
        let types = builder.build();
        let i32 = types.resolve_name("i32").unwrap();
        assert_eq!(
            types
                .resolve_binary(
                    BinaryOperator::Add,
                    TypeConstraint::Known(i32),
                    TypeConstraint::Literal(LiteralKind::Integer),
                    None,
                )
                .unwrap()
                .result,
            i32
        );
    }
}
