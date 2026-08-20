use crate::{layout_of, AbiTarget, AbiType, CallingConvention, LayoutError, ScalarType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterClass {
    Integer,
    Float,
    Vector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassMode {
    Ignore,
    Direct(Vec<RegisterClass>),
    Indirect { alignment: u32, by_value: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedValue {
    pub ty: AbiType,
    pub mode: PassMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiSignature {
    pub convention: CallingConvention,
    pub parameters: Vec<AbiType>,
    pub result: AbiType,
    pub variadic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedSignature {
    pub convention: CallingConvention,
    pub arguments: Vec<ClassifiedValue>,
    pub result: ClassifiedValue,
    pub variadic: bool,
}

pub fn classify_signature(
    signature: &AbiSignature,
    target: &AbiTarget,
) -> Result<ClassifiedSignature, LayoutError> {
    let convention = target.resolve_convention(signature.convention);
    let arguments = signature
        .parameters
        .iter()
        .map(|ty| classify(ty, convention, target, false))
        .collect::<Result<Vec<_>, _>>()?;
    let result = classify(&signature.result, convention, target, true)?;
    Ok(ClassifiedSignature {
        convention,
        arguments,
        result,
        variadic: signature.variadic,
    })
}

fn classify(
    ty: &AbiType,
    convention: CallingConvention,
    target: &AbiTarget,
    result: bool,
) -> Result<ClassifiedValue, LayoutError> {
    let layout = layout_of(ty, &target.data_layout)?;
    let mode = if matches!(ty, AbiType::Void) || layout.size == 0 {
        PassMode::Ignore
    } else if let AbiType::Scalar(scalar) = ty {
        PassMode::Direct(vec![match scalar {
            ScalarType::Float { .. } => RegisterClass::Float,
            ScalarType::Integer { .. } | ScalarType::Boolean => RegisterClass::Integer,
        }])
    } else if matches!(ty, AbiType::Pointer { .. } | AbiType::Function(_)) {
        PassMode::Direct(vec![RegisterClass::Integer])
    } else {
        classify_aggregate(ty, &layout, convention, result)
    };
    Ok(ClassifiedValue {
        ty: ty.clone(),
        mode,
    })
}

fn classify_aggregate(
    ty: &AbiType,
    layout: &crate::Layout,
    convention: CallingConvention,
    result: bool,
) -> PassMode {
    let direct_limit = match convention {
        CallingConvention::Win64 => 8,
        CallingConvention::SysV64 | CallingConvention::Aapcs64 | CallingConvention::Rust => 16,
        CallingConvention::C | CallingConvention::System => 8,
    };
    if layout.size > direct_limit
        || (convention == CallingConvention::Win64 && !matches!(layout.size, 1 | 2 | 4 | 8))
    {
        return PassMode::Indirect {
            alignment: layout.alignment,
            by_value: !result,
        };
    }
    let chunks = usize::try_from(layout.size.div_ceil(8))
        .unwrap_or(usize::MAX)
        .max(1);
    let class = if matches!(
        convention,
        CallingConvention::SysV64 | CallingConvention::Aapcs64
    ) && homogeneous_float_aggregate(ty)
    {
        RegisterClass::Float
    } else {
        RegisterClass::Integer
    };
    PassMode::Direct(vec![class; chunks])
}

fn homogeneous_float_aggregate(ty: &AbiType) -> bool {
    match ty {
        AbiType::Scalar(ScalarType::Float { .. }) => true,
        AbiType::Array { element, length } => *length > 0 && homogeneous_float_aggregate(element),
        AbiType::Record(record) => {
            !record.fields.is_empty()
                && record
                    .fields
                    .iter()
                    .all(|field| homogeneous_float_aggregate(&field.ty))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{layout_of, Field, LayoutKind, RecordRepresentation, RecordType};
    use severian_target::TargetSpec;

    fn target(triple: &str) -> AbiTarget {
        AbiTarget::derive(&TargetSpec::new(triple))
    }

    fn record(length: u64) -> AbiType {
        AbiType::Record(RecordType {
            name: None,
            fields: vec![Field {
                name: "bytes".into(),
                ty: AbiType::Array {
                    element: Box::new(AbiType::integer(8, false)),
                    length,
                },
            }],
            representation: RecordRepresentation::C,
        })
    }

    #[test]
    fn target_layout_places_padding_and_tail_padding() {
        let target = target("x86_64-unknown-linux");
        let record = AbiType::Record(RecordType {
            name: Some("Example".into()),
            fields: vec![
                Field {
                    name: "flag".into(),
                    ty: AbiType::integer(8, false),
                },
                Field {
                    name: "value".into(),
                    ty: AbiType::integer(32, true),
                },
            ],
            representation: RecordRepresentation::C,
        });
        let layout = layout_of(&record, &target.data_layout).unwrap();
        assert_eq!((layout.size, layout.alignment), (8, 4));
        let LayoutKind::Record { fields } = layout.kind else {
            unreachable!()
        };
        assert_eq!(fields[1].offset, 4);
    }

    #[test]
    fn sysv_and_win64_classify_large_records_indirectly() {
        for convention in [CallingConvention::SysV64, CallingConvention::Win64] {
            let target = target("x86_64-unknown-linux");
            let signature = AbiSignature {
                convention,
                parameters: vec![record(24)],
                result: record(24),
                variadic: false,
            };
            let classified = classify_signature(&signature, &target).unwrap();
            assert!(matches!(
                classified.arguments[0].mode,
                PassMode::Indirect { by_value: true, .. }
            ));
            assert!(matches!(
                classified.result.mode,
                PassMode::Indirect {
                    by_value: false,
                    ..
                }
            ));
        }
    }

    #[test]
    fn system_convention_resolves_from_target() {
        let target = target("x86_64-pc-windows");
        let signature = AbiSignature {
            convention: CallingConvention::System,
            parameters: vec![AbiType::integer(32, true)],
            result: AbiType::Void,
            variadic: false,
        };
        assert_eq!(
            classify_signature(&signature, &target).unwrap().convention,
            CallingConvention::Win64
        );
    }

    #[test]
    fn sysv_classifies_homogeneous_float_aggregates_in_float_registers() {
        let target = target("x86_64-unknown-linux");
        let pair = AbiType::Record(RecordType {
            name: None,
            fields: vec![
                Field {
                    name: "x".into(),
                    ty: AbiType::float(32),
                },
                Field {
                    name: "y".into(),
                    ty: AbiType::float(32),
                },
            ],
            representation: RecordRepresentation::C,
        });
        let signature = AbiSignature {
            convention: CallingConvention::SysV64,
            parameters: vec![pair],
            result: AbiType::Void,
            variadic: false,
        };
        assert_eq!(
            classify_signature(&signature, &target).unwrap().arguments[0].mode,
            PassMode::Direct(vec![RegisterClass::Float])
        );
    }
}
