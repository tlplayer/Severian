use crate::{
    validate_function, FfiError, ForeignFunction, ForeignModule, ForeignParameter, ForeignTypeRef,
    ParameterMode, ValueContract,
};
use severian_abi::{
    AbiSignature, AbiTarget, AbiType, Field, RecordRepresentation, RecordType, Symbol,
};
use severian_universal::{FloatFormat, IntegerWidth, PrimitiveRepresentation, TypeContext};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conversion {
    Direct,
    Boolean,
    Utf8View,
    BytesView,
    OpaquePointer,
    OutPointer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredParameter {
    pub name: String,
    pub abi_type: AbiType,
    pub conversion: Conversion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryPlan {
    pub provider: Option<String>,
    pub symbol: Symbol,
    pub signature: AbiSignature,
    pub parameters: Vec<LoweredParameter>,
    pub result_type: AbiType,
    pub result_conversion: Conversion,
}

pub fn lower_function(
    function: &ForeignFunction,
    module: &ForeignModule,
    types: &TypeContext,
    target: &AbiTarget,
) -> Result<BoundaryPlan, FfiError> {
    validate_function(function, module, target)?;
    let parameters = function
        .parameters
        .iter()
        .map(|parameter| lower_parameter(parameter, module, types, target))
        .collect::<Result<Vec<_>, _>>()?;
    let (result_type, result_conversion) =
        lower_contract(&function.result, module, types, target, true)?;
    let signature = AbiSignature {
        convention: function.abi.convention(),
        parameters: parameters
            .iter()
            .map(|parameter| parameter.abi_type.clone())
            .collect(),
        result: result_type.clone(),
        variadic: function.variadic,
    };
    Ok(BoundaryPlan {
        provider: function.provider.clone(),
        symbol: function.symbol.clone(),
        signature,
        parameters,
        result_type,
        result_conversion,
    })
}

fn lower_parameter(
    parameter: &ForeignParameter,
    module: &ForeignModule,
    types: &TypeContext,
    target: &AbiTarget,
) -> Result<LoweredParameter, FfiError> {
    let (mut abi_type, mut conversion) =
        lower_contract(&parameter.contract, module, types, target, false)?;
    if matches!(parameter.mode, ParameterMode::Out | ParameterMode::InOut) {
        abi_type = AbiType::pointer_to(abi_type, true);
        conversion = Conversion::OutPointer;
    }
    Ok(LoweredParameter {
        name: parameter.name.clone(),
        abi_type,
        conversion,
    })
}

fn lower_contract(
    contract: &ValueContract,
    module: &ForeignModule,
    types: &TypeContext,
    target: &AbiTarget,
    result: bool,
) -> Result<(AbiType, Conversion), FfiError> {
    let (abi_type, conversion) = lower_type_ref(&contract.ty, module, types, target, result)?;
    if contract.nullable && !matches!(abi_type, AbiType::Pointer { .. }) {
        return Err(FfiError::InvalidOwnership(
            "only pointer representations may be nullable".into(),
        ));
    }
    Ok((abi_type, conversion))
}

fn lower_type_ref(
    ty: &ForeignTypeRef,
    module: &ForeignModule,
    types: &TypeContext,
    target: &AbiTarget,
    result: bool,
) -> Result<(AbiType, Conversion), FfiError> {
    match ty {
        ForeignTypeRef::Severian(id) => lower_semantic(*id, types, target, result),
        ForeignTypeRef::External(name) => module
            .type_declaration(name)
            .map(|declaration| (declaration.representation.clone(), Conversion::Direct))
            .ok_or_else(|| FfiError::UnknownExternalType(name.clone())),
        ForeignTypeRef::Pointer { pointee, mutable } => {
            let (pointee, _) = lower_type_ref(pointee, module, types, target, false)?;
            Ok((AbiType::pointer_to(pointee, *mutable), Conversion::Direct))
        }
    }
}

fn lower_semantic(
    id: severian_universal::TypeId,
    types: &TypeContext,
    target: &AbiTarget,
    result: bool,
) -> Result<(AbiType, Conversion), FfiError> {
    let primitive = types.primitive(id).ok_or(FfiError::NotPrimitive(id))?;
    Ok(match primitive.representation {
        PrimitiveRepresentation::Integer { bits, signed } => (
            AbiType::integer(
                match bits {
                    IntegerWidth::Fixed(bits) => bits,
                    IntegerWidth::Machine => target.data_layout.machine_integer_bits,
                },
                signed,
            ),
            Conversion::Direct,
        ),
        PrimitiveRepresentation::PointerInteger { signed } => (
            AbiType::integer(
                u16::try_from(target.data_layout.pointer.size * 8).unwrap_or(u16::MAX),
                signed,
            ),
            Conversion::Direct,
        ),
        PrimitiveRepresentation::Float { format } => (
            match format {
                FloatFormat::Float8E4M3Fn => AbiType::float8_e4m3fn(),
                FloatFormat::Float8E5M2 => AbiType::float8_e5m2(),
                FloatFormat::Ieee(bits) => AbiType::float(bits),
                FloatFormat::BrainFloat16 => AbiType::bfloat16(),
                FloatFormat::Machine => AbiType::float(target.data_layout.machine_float_bits),
            },
            Conversion::Direct,
        ),
        PrimitiveRepresentation::Boolean => (
            AbiType::Scalar(severian_abi::ScalarType::Boolean),
            Conversion::Boolean,
        ),
        PrimitiveRepresentation::Character => (AbiType::integer(32, false), Conversion::Direct),
        PrimitiveRepresentation::String => (view_type("StringView", target), Conversion::Utf8View),
        PrimitiveRepresentation::Bytes => (view_type("BytesView", target), Conversion::BytesView),
        PrimitiveRepresentation::None | PrimitiveRepresentation::Unit if result => {
            (AbiType::Void, Conversion::Direct)
        }
        PrimitiveRepresentation::None | PrimitiveRepresentation::Unit => {
            return Err(FfiError::UnsupportedRepresentation(
                "none/unit cannot be passed as an argument".into(),
            ))
        }
        PrimitiveRepresentation::Arguments => {
            return Err(FfiError::UnsupportedRepresentation(
                "program arguments cannot cross an external function boundary".into(),
            ))
        }
    })
}

fn view_type(name: &str, target: &AbiTarget) -> AbiType {
    AbiType::Record(RecordType {
        name: Some(name.into()),
        fields: vec![
            Field {
                name: "data".into(),
                ty: AbiType::pointer_to(AbiType::integer(8, false), false),
            },
            Field {
                name: "length".into(),
                ty: AbiType::integer(
                    u16::try_from(target.data_layout.pointer.size * 8).unwrap_or(u16::MAX),
                    false,
                ),
            },
        ],
        representation: RecordRepresentation::C,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AbiSelection, ForeignParameter, Lifetime, Ownership};
    use severian_abi::CallingConvention;
    use severian_target::TargetSpec;
    use severian_universal::{
        PrimitiveCategory, PrimitiveRepresentation, TypeContext, TypeContextBuilder,
    };

    fn primitive_types() -> (
        TypeContext,
        severian_universal::TypeId,
        severian_universal::TypeId,
    ) {
        let mut types = TypeContextBuilder::new();
        let i32_id = types.register_declaration("core.i32", "i32").unwrap();
        types
            .define_primitive(
                i32_id,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(32),
                    signed: true,
                },
                false,
            )
            .unwrap();
        let string_id = types.register_declaration("core.string", "string").unwrap();
        types
            .define_primitive(
                string_id,
                PrimitiveCategory::Text,
                PrimitiveRepresentation::String,
                false,
            )
            .unwrap();
        (types.build(), i32_id, string_id)
    }

    fn contract(ty: ForeignTypeRef, ownership: Ownership) -> ValueContract {
        ValueContract {
            ty,
            ownership,
            nullable: false,
        }
    }

    fn target() -> AbiTarget {
        AbiTarget::derive(&TargetSpec::new("x86_64-unknown-linux"))
    }

    #[test]
    fn lowers_semantic_values_and_classifies_the_call() {
        let (types, i32_id, string_id) = primitive_types();
        let function = ForeignFunction {
            name: "length".into(),
            provider: Some("text-runtime".into()),
            symbol: Symbol::imported_function("sev_length").unwrap(),
            parameters: vec![ForeignParameter {
                name: "text".into(),
                contract: contract(
                    ForeignTypeRef::Severian(string_id),
                    Ownership::Borrowed(Lifetime::Call),
                ),
                mode: ParameterMode::In,
            }],
            result: contract(ForeignTypeRef::Severian(i32_id), Ownership::Copy),
            abi: AbiSelection::C,
            variadic: false,
        };
        let plan = lower_function(&function, &ForeignModule::default(), &types, &target()).unwrap();
        assert_eq!(plan.signature.convention, CallingConvention::C);
        assert_eq!(
            severian_abi::classify_signature(&plan.signature, &target())
                .unwrap()
                .convention,
            CallingConvention::SysV64
        );
        assert_eq!(plan.parameters[0].conversion, Conversion::Utf8View);
    }

    #[test]
    fn rejects_call_lifetime_borrowed_returns() {
        let (types, _, string_id) = primitive_types();
        let function = ForeignFunction {
            name: "bad".into(),
            provider: None,
            symbol: Symbol::imported_function("bad").unwrap(),
            parameters: vec![],
            result: contract(
                ForeignTypeRef::Severian(string_id),
                Ownership::Borrowed(Lifetime::Call),
            ),
            abi: AbiSelection::C,
            variadic: false,
        };
        assert_eq!(
            lower_function(&function, &ForeignModule::default(), &types, &target()).unwrap_err(),
            FfiError::ReturnCannotBorrowCall
        );
    }

    #[test]
    fn preserves_bfloat_as_a_distinct_abi_format() {
        let mut types = TypeContextBuilder::new();
        let id = types.register_declaration("core.bf16", "bf16").unwrap();
        types
            .define_primitive(
                id,
                PrimitiveCategory::Float,
                PrimitiveRepresentation::Float {
                    format: FloatFormat::BrainFloat16,
                },
                false,
            )
            .unwrap();
        let types = types.build();
        assert_eq!(
            lower_semantic(id, &types, &target(), false).unwrap().0,
            AbiType::bfloat16()
        );
    }
}
