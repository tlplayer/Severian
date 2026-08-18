use severian_abi::*;

fn register_view(registry: &mut AbiRegistry) {
    let t = SchemaParamId::new(0);
    let space = SchemaParamId::new(1);

    registry.register_schema(AbiSchema::new(
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
    )).unwrap();
}

/// This test intentionally defines a tensor-like descriptor outside the ABI
/// crate. It composes the generic View schema instead of adding Tensor/Shape/
/// Strides variants to `AbiType`.
fn register_tensor_descriptor(registry: &mut AbiRegistry) {
    let t = SchemaParamId::new(0);
    let space = SchemaParamId::new(1);

    let metadata_view = |element: AbiType| {
        AbiTypeExpr::apply(
            AbiSchemaId::new("core.View"),
            vec![
                AbiArgumentExpr::Type(AbiTypeExpr::concrete(element)),
                AbiArgumentExpr::AddressSpace(AbiAddressSpaceExpr::default_space()),
            ],
        )
    };

    registry.register_schema(AbiSchema::new(
        AbiSchemaId::new("tensor.DenseDescriptor"),
        vec![
            SchemaParam::ty(0, "T"),
            SchemaParam::address_space(1, "Space"),
        ],
        AbiTypeExpr::Record(RecordTypeExpr {
            id: RecordId::new("DenseTensorDescriptor"),
            repr: RecordRepr::C,
            fields: vec![
                RecordFieldExpr {
                    name: "data".into(),
                    ty: AbiTypeExpr::Pointer(
                        PointerTypeExpr::new(AbiTypeExpr::TypeParam(t))
                            .mutable()
                            .in_address_space(AbiAddressSpaceExpr::Param(space)),
                    ),
                },
                RecordFieldExpr {
                    name: "shape".into(),
                    ty: metadata_view(AbiType::usize()),
                },
                RecordFieldExpr {
                    name: "strides".into(),
                    ty: metadata_view(AbiType::isize()),
                },
            ],
        }),
    )).unwrap();
}

#[test]
fn tensor_descriptor_is_composed_from_ordinary_abi_schemas() {
    let mut registry = AbiRegistry::new();
    register_view(&mut registry);
    register_tensor_descriptor(&mut registry);

    let tensor = registry.instantiate(
        &AbiSchemaId::new("tensor.DenseDescriptor"),
        vec![
            AbiArgument::Type(AbiType::bf16()),
            AbiArgument::AddressSpace(AddressSpaceId::new("gpu.global")),
        ],
    ).unwrap();

    let AbiType::Record(record) = tensor.ty else {
        panic!("tensor descriptor should expand to a record");
    };
    let AbiType::Pointer(data) = &record.fields[0].ty else {
        panic!("tensor data should be a pointer");
    };
    let AbiType::Record(shape_view) = &record.fields[1].ty else {
        panic!("shape should be an expanded View[usize]");
    };

    assert_eq!(*data.pointee, AbiType::bf16());
    assert_eq!(data.address_space, AddressSpaceId::new("gpu.global"));
    assert_eq!(shape_view.fields.len(), 2);
}

#[test]
fn target_can_define_device_pointer_layout_without_knowing_the_runtime() {
    let target = TargetDataLayout::new(Endianness::Little, Layout::new(8, 8))
        .with_pointer_layout(AddressSpaceId::new("gpu.global"), Layout::new(8, 8));

    let pointer = AbiType::Pointer(
        PointerType::new(AbiType::u8())
            .in_address_space(AddressSpaceId::new("gpu.global")),
    );

    assert_eq!(layout_of(&pointer, &target).unwrap(), Layout::new(8, 8));
}
