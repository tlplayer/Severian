use severian_abi::*;

fn view_schema() -> AbiSchema {
    let t = SchemaParamId::new(0);
    let space = SchemaParamId::new(1);

    AbiSchema::new(
        AbiSchemaId::new("core.View"),
        vec![
            SchemaParam::ty(0, "T"),
            SchemaParam::address_space(1, "Space"),
        ],
        AbiTypeExpr::Record(RecordTypeExpr {
            id: RecordId::new("View"),
            repr: RecordRepr::C,
            fields: vec![
                RecordFieldExpr {
                    name: "data".into(),
                    ty: AbiTypeExpr::Pointer(
                        PointerTypeExpr::new(AbiTypeExpr::TypeParam(t))
                            .in_address_space(AbiAddressSpaceExpr::Param(space)),
                    ),
                },
                RecordFieldExpr {
                    name: "length".into(),
                    ty: AbiTypeExpr::concrete(AbiType::usize()),
                },
            ],
        }),
    )
}

#[test]
fn generic_view_instantiates_without_view_becoming_an_abi_variant() {
    let mut registry = AbiRegistry::with_builtin_abis();
    registry.register_schema(view_schema()).unwrap();

    let instance = registry.instantiate(
        &AbiSchemaId::new("core.View"),
        vec![
            AbiArgument::Type(AbiType::f32()),
            AbiArgument::AddressSpace(AddressSpaceId::new("device")),
        ],
    ).unwrap();

    let AbiType::Record(view) = instance.ty else {
        panic!("View should expand to an ordinary record");
    };
    let AbiType::Pointer(data) = &view.fields[0].ty else {
        panic!("View.data should be a pointer");
    };

    assert_eq!(*data.pointee, AbiType::f32());
    assert_eq!(data.address_space, AddressSpaceId::new("device"));
}

#[test]
fn const_parameters_support_generic_fixed_arrays() {
    let t = SchemaParamId::new(0);
    let n = SchemaParamId::new(1);
    let mut registry = AbiRegistry::new();
    registry.register_schema(AbiSchema::new(
        AbiSchemaId::new("core.Fixed"),
        vec![SchemaParam::ty(0, "T"), SchemaParam::constant(1, "N")],
        AbiTypeExpr::Array(ArrayTypeExpr {
            element: Box::new(AbiTypeExpr::TypeParam(t)),
            length: AbiConstExpr::Param(n),
        }),
    )).unwrap();

    let instance = registry.instantiate(
        &AbiSchemaId::new("core.Fixed"),
        vec![AbiArgument::Type(AbiType::u16()), AbiArgument::Const(16)],
    ).unwrap();

    assert_eq!(
        instance.ty,
        AbiType::Array(ArrayType { element: Box::new(AbiType::u16()), length: 16 })
    );
}
