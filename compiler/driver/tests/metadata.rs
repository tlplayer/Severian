use severian_hir::{FunctionId, TypeKind};

#[test]
fn compilation_carries_structural_metadata_without_executing_it() {
    let source = concat!(
        "native(\"load_values\") def loadValues(values: list[int]) -> Result[list[int], string]\n",
        "\n",
        "def main():\n",
        "    print(\"wired\")\n",
    );

    let compilation = severian_driver::compile_source(source).unwrap();

    assert_eq!(compilation.hir.metadata.sources.files().len(), 1);
    assert_eq!(compilation.hir.metadata.sources.files()[0].source, source);
    let signature = &compilation.hir.metadata.functions[&FunctionId::from_name("loadValues")];
    assert!(matches!(
        compilation.hir.metadata.types.get(signature.parameters[0]),
        Some(TypeKind::List(_))
    ));
    assert!(matches!(
        compilation.hir.metadata.types.get(signature.returns),
        Some(TypeKind::Result { .. })
    ));
    assert_eq!(
        compilation.mir.metadata(),
        &compilation.optimized_hir.metadata
    );
}
