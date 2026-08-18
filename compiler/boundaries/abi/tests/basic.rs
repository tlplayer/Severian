use severian_abi::*;

fn lp64() -> TargetDataLayout {
    TargetDataLayout::new(Endianness::Little, Layout::new(8, 8))
        .with_integer_alignment(8, 1)
        .with_integer_alignment(16, 2)
        .with_integer_alignment(32, 4)
        .with_integer_alignment(64, 8)
        .with_float_alignment(16, 2)
        .with_float_alignment(32, 4)
        .with_float_alignment(64, 8)
}

#[test]
fn out_is_call_mode_not_wrapper_type() {
    let signature = AbiSignature::new(
        AbiId::new("c"),
        vec![AbiParameter::output("written", AbiValue::copy(AbiType::usize()))],
        AbiValue::copy(AbiType::i32()),
    );

    assert_eq!(signature.parameters[0].mode, ParameterMode::Out);
    assert_eq!(signature.parameters[0].value.ty, AbiType::usize());
}

#[test]
fn opaque_is_legal_only_behind_pointer() {
    let opaque = AbiType::Opaque(OpaqueId::new("FILE"));
    assert!(matches!(
        validate_type(&opaque, Position::Parameter),
        Err(AbiError::OpaqueByValue(_))
    ));

    let pointer = AbiType::pointer_to(opaque);
    assert!(validate_type(&pointer, Position::Parameter).is_ok());
}

#[test]
fn c_record_layout_is_target_driven() {
    let record = RecordType {
        id: RecordId::new("Pair"),
        repr: RecordRepr::C,
        fields: vec![
            RecordField { name: "tag".into(), ty: AbiType::u8() },
            RecordField { name: "value".into(), ty: AbiType::u32() },
        ],
    };

    let layout = layout_record(&record, &lp64()).unwrap();
    assert_eq!(layout.fields[0].offset, 0);
    assert_eq!(layout.fields[1].offset, 4);
    assert_eq!(layout.layout, Layout::new(8, 4));
}
