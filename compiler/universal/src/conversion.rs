use crate::{
    FloatFormat, IntegerWidth, PrimitiveCategory, PrimitiveRepresentation, TypeContext, TypeId,
};

/// The semantic contract used to convert one numeric value into another.
///
/// The ordering is intentional: later kinds permit progressively more change
/// to the source value. It is used when an explicit constructor mode is
/// requested (for example, `i8(value, lossy)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionKind {
    Identity,
    Promote,
    Checked,
    Lossy,
}

impl ConversionKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "identity" => Some(Self::Identity),
            "promote" => Some(Self::Promote),
            "checked" => Some(Self::Checked),
            "lossy" => Some(Self::Lossy),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Promote => "promote",
            Self::Checked => "checked",
            Self::Lossy => "lossy",
        }
    }

    /// Returns whether this explicitly selected mode can implement a
    /// conversion whose default safety requirement is `required`.
    pub const fn permits(self, required: Self) -> bool {
        self as u8 >= required as u8
    }
}

/// A constructor accepted by one target type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constructor {
    pub source: TypeId,
    pub kind: ConversionKind,
}

/// A resolved conversion attached to an expression and preserved through IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conversion {
    pub from: TypeId,
    pub to: TypeId,
    pub kind: ConversionKind,
}

impl Conversion {
    pub const fn with_kind(self, kind: ConversionKind) -> Option<Self> {
        if kind.permits(self.kind) {
            Some(Self { kind, ..self })
        } else {
            None
        }
    }
}

impl TypeContext {
    /// Resolves the default numeric conversion between two universal types.
    pub fn numeric_conversion(&self, from: TypeId, to: TypeId) -> Option<Conversion> {
        let source = self.primitive(from)?;
        let target = self.primitive(to)?;
        let source_numeric = numeric_category(source.category);
        let target_numeric = numeric_category(target.category);
        if !source_numeric || !target_numeric {
            return None;
        }
        let kind = if from == to {
            ConversionKind::Identity
        } else if source.category == PrimitiveCategory::Measured
            || target.category == PrimitiveCategory::Measured
        {
            if source.category == PrimitiveCategory::Measured
                && target.category == PrimitiveCategory::Measured
            {
                return None;
            }
            // Entering or leaving a dimension is always explicit, even when
            // both values share one physical floating-point representation.
            ConversionKind::Lossy
        } else {
            classify(source.representation, target.representation)?
        };
        Some(Conversion { from, to, kind })
    }

    /// Returns the constructor set exposed by a numeric target type.
    pub fn numeric_constructors(&self, target: TypeId) -> Vec<Constructor> {
        let mut constructors = self
            .definitions()
            .filter_map(|definition| {
                self.numeric_conversion(definition.id, target)
                    .map(|conversion| Constructor {
                        source: conversion.from,
                        kind: conversion.kind,
                    })
            })
            .collect::<Vec<_>>();
        constructors.sort();
        constructors.dedup();
        constructors
    }

    /// Stable overload-ranking cost derived from the same conversion policy.
    pub fn numeric_conversion_cost(&self, from: TypeId, to: TypeId) -> Option<u32> {
        let conversion = self.numeric_conversion(from, to)?;
        let source = self.primitive(from)?.representation;
        let target = self.primitive(to)?.representation;
        Some(match conversion.kind {
            ConversionKind::Identity => 0,
            ConversionKind::Promote => promotion_cost(source, target),
            ConversionKind::Checked => 1_000,
            ConversionKind::Lossy => 2_000,
        })
    }
}

const fn numeric_category(category: PrimitiveCategory) -> bool {
    matches!(
        category,
        PrimitiveCategory::Integer | PrimitiveCategory::Float | PrimitiveCategory::Measured
    )
}

fn classify(
    source: PrimitiveRepresentation,
    target: PrimitiveRepresentation,
) -> Option<ConversionKind> {
    use ConversionKind::{Checked, Lossy, Promote};
    use PrimitiveRepresentation::{Float, Integer, PointerInteger};

    match (source, target) {
        (
            Integer {
                bits: source_bits,
                signed: source_signed,
            },
            Integer {
                bits: target_bits,
                signed: target_signed,
            },
        ) => Some(
            if integer_range_fits(source_bits, source_signed, target_bits, target_signed) {
                Promote
            } else {
                Checked
            },
        ),
        (
            PointerInteger {
                signed: source_signed,
            },
            PointerInteger {
                signed: target_signed,
            },
        ) => Some(if source_signed == target_signed {
            Promote
        } else {
            Checked
        }),
        (Integer { .. }, PointerInteger { .. }) | (PointerInteger { .. }, Integer { .. }) => {
            Some(Checked)
        }

        // Integer-to-float is the language's ordinary arithmetic promotion.
        // Backends still choose the concrete signed/unsigned instruction.
        (Integer { .. } | PointerInteger { .. }, Float { .. }) => Some(Promote),
        (Float { .. }, Integer { .. } | PointerInteger { .. }) => Some(Lossy),

        (
            Float {
                format: source_format,
            },
            Float {
                format: target_format,
            },
        ) => Some(if float_format_fits(source_format, target_format) {
            Promote
        } else {
            Lossy
        }),
        _ => None,
    }
}

fn integer_range_fits(
    source: IntegerWidth,
    source_signed: bool,
    target: IntegerWidth,
    target_signed: bool,
) -> bool {
    match (source, target) {
        (IntegerWidth::Fixed(source), IntegerWidth::Fixed(target)) => {
            if source_signed == target_signed {
                source <= target
            } else if source_signed {
                false
            } else {
                source < target
            }
        }
        (IntegerWidth::Machine, IntegerWidth::Machine) => source_signed == target_signed,
        _ => false,
    }
}

fn float_format_fits(source: FloatFormat, target: FloatFormat) -> bool {
    match (float_shape(source), float_shape(target)) {
        (Some((source_exponent, source_precision)), Some((target_exponent, target_precision))) => {
            source_exponent <= target_exponent && source_precision <= target_precision
        }
        (None, None) => true,
        _ => false,
    }
}

fn promotion_cost(source: PrimitiveRepresentation, target: PrimitiveRepresentation) -> u32 {
    match (source, target) {
        (
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(source),
                ..
            },
            PrimitiveRepresentation::Integer {
                bits: IntegerWidth::Fixed(target),
                ..
            },
        ) => u32::from(target.saturating_sub(source)) + 1,
        (
            PrimitiveRepresentation::Float {
                format: source_format,
            },
            PrimitiveRepresentation::Float {
                format: target_format,
            },
        ) => float_shape(source_format)
            .zip(float_shape(target_format))
            .map_or(
                1,
                |((source_exponent, source_precision), (target_exponent, target_precision))| {
                    u32::from(target_exponent.saturating_sub(source_exponent))
                        + u32::from(target_precision.saturating_sub(source_precision))
                        + 1
                },
            ),
        (
            PrimitiveRepresentation::Integer { .. }
            | PrimitiveRepresentation::PointerInteger { .. },
            PrimitiveRepresentation::Float { .. },
        ) => 100,
        _ => 1,
    }
}

/// Returns `(exponent bits, significand precision)` for known binary formats.
fn float_shape(format: FloatFormat) -> Option<(u16, u16)> {
    match format {
        FloatFormat::Float8E4M3Fn => Some((4, 4)),
        FloatFormat::Float8E5M2 => Some((5, 3)),
        FloatFormat::Ieee(16) => Some((5, 11)),
        FloatFormat::BrainFloat16 => Some((8, 8)),
        FloatFormat::Ieee(32) => Some((8, 24)),
        FloatFormat::Ieee(64) => Some((11, 53)),
        FloatFormat::Ieee(128) => Some((15, 113)),
        FloatFormat::Ieee(_) | FloatFormat::Machine => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{install_primitives, TypeContextBuilder};

    fn types() -> TypeContext {
        let mut types = TypeContextBuilder::new();
        install_primitives(&mut types).unwrap();
        types.build()
    }

    #[test]
    fn numeric_defaults_cover_width_sign_and_category_changes() {
        let types = types();
        let id = |name| types.resolve_name(name).unwrap();
        let kind = |from, to| types.numeric_conversion(id(from), id(to)).unwrap().kind;

        assert_eq!(kind("i32", "i32"), ConversionKind::Identity);
        assert_eq!(kind("i32", "i64"), ConversionKind::Promote);
        assert_eq!(kind("u32", "i64"), ConversionKind::Promote);
        assert_eq!(kind("i64", "i32"), ConversionKind::Checked);
        assert_eq!(kind("i32", "u32"), ConversionKind::Checked);
        assert_eq!(kind("i32", "f32"), ConversionKind::Promote);
        assert_eq!(kind("f32", "i32"), ConversionKind::Lossy);
        assert_eq!(kind("f16", "f32"), ConversionKind::Promote);
        assert_eq!(kind("f8e4m3fn", "f16"), ConversionKind::Promote);
        assert_eq!(kind("f8e5m2", "f16"), ConversionKind::Promote);
        assert_eq!(kind("f8e4m3fn", "f8e5m2"), ConversionKind::Lossy);
        assert_eq!(kind("f8e5m2", "f8e4m3fn"), ConversionKind::Lossy);
        assert_eq!(kind("bf16", "f16"), ConversionKind::Lossy);
        assert_eq!(kind("f64", "f32"), ConversionKind::Lossy);
        assert_eq!(kind("f64", "f128"), ConversionKind::Promote);
        assert_eq!(kind("f128", "f64"), ConversionKind::Lossy);
        assert_eq!(kind("int", "data_size"), ConversionKind::Lossy);
        assert_eq!(kind("duration", "float"), ConversionKind::Lossy);
        assert!(types
            .numeric_conversion(id("duration"), id("data_size"))
            .is_none());
    }

    #[test]
    fn target_constructor_set_uses_the_same_conversion_policy() {
        let types = types();
        let i32 = types.resolve_name("i32").unwrap();
        let constructors = types.numeric_constructors(i32);
        let from = |name| {
            constructors
                .iter()
                .find(|constructor| constructor.source == types.resolve_name(name).unwrap())
                .unwrap()
                .kind
        };

        assert_eq!(from("i16"), ConversionKind::Promote);
        assert_eq!(from("i64"), ConversionKind::Checked);
        assert_eq!(from("f64"), ConversionKind::Lossy);
        assert!(constructors
            .iter()
            .all(|constructor| constructor.source != types.resolve_name("string").unwrap()));
    }

    #[test]
    fn scalar_policy_is_total_for_every_registered_tensor_element_shape() {
        let types = types();
        let numeric = types
            .definitions()
            .filter(|definition| {
                types.primitive(definition.id).is_some_and(|primitive| {
                    matches!(
                        primitive.category,
                        PrimitiveCategory::Integer | PrimitiveCategory::Float
                    )
                })
            })
            .map(|definition| definition.id)
            .collect::<Vec<_>>();

        assert_eq!(numeric.len(), 21);
        for source in &numeric {
            for target in &numeric {
                assert!(
                    types.numeric_conversion(*source, *target).is_some(),
                    "missing numeric conversion from {source:?} to {target:?}"
                );
            }
        }
    }

    #[test]
    fn an_explicit_mode_cannot_promise_more_than_the_conversion_supports() {
        let types = types();
        let conversion = types
            .numeric_conversion(
                types.resolve_name("f64").unwrap(),
                types.resolve_name("i32").unwrap(),
            )
            .unwrap();
        assert_eq!(conversion.with_kind(ConversionKind::Promote), None);
        assert_eq!(
            conversion.with_kind(ConversionKind::Lossy).unwrap().kind,
            ConversionKind::Lossy
        );
    }
}
