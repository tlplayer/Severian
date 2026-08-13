use severian_hir::{
    DefinitionId, Expression, FunctionId, Instruction, TaskPlacement, TensorDimension,
    TensorElementType, TypeDefinitionId, TypeKind, ValueType, VariantId,
};
use severian_lexer::lex;
use severian_package::PackageInterface;
use severian_parser::parse;
use severian_semantic::{
    analyze, analyze_with_interfaces, analyze_with_packages, attach_module_metadata,
    attach_module_metadata_with_packages,
};
use std::path::PathBuf;

#[test]
fn resolves_print_and_lowers_hello_to_hir() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    let Instruction::Print(value) = &hir.main().unwrap().instructions[0] else {
        panic!("expected print instruction");
    };
    assert_eq!(value.ty(), Some(ValueType::String));
    assert!(value.hir_id().is_some());
    assert_eq!(value.kind(), &Expression::String("hello, severian".into()));
}

#[test]
fn list_addition_has_list_type() {
    let ast =
        parse(&lex("def combined() -> list[int]:\n    return [1, 2] + [3]\n").unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected a return value");
    };
    assert_eq!(value.ty(), Some(ValueType::List));
}

#[test]
fn bare_return_from_unit_result_produces_ok_variant() {
    let source = "def save() -> Result[unit, string]:\n    return\n";
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected an ok return value");
    };
    let Expression::Variant { name, fields, .. } = value.kind() else {
        panic!("expected a Result variant");
    };
    assert_eq!(name, "ok");
    assert!(fields.is_empty());
}

#[test]
fn assignment_propagates_results_while_try_equal_captures_them() {
    let source = concat!(
        "def read() -> Result[int, string]:\n",
        "    return 42\n",
        "\n",
        "def stable() -> Result[int, string]:\n",
        "    value = read()\n",
        "    return value\n",
        "\n",
        "def changeable() -> Result[int, string]:\n",
        "    value := read()\n",
        "    value += 1\n",
        "    return value\n",
        "\n",
        "def handled() -> int:\n",
        "    outcome ?= read()\n",
        "    switch outcome:\n",
        "        ok value:\n",
        "            return value\n",
        "        failure _:\n",
        "            return 0\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    assert!(matches!(
        hir.functions[1].instructions[0],
        Instruction::TryLet { .. }
    ));
    assert!(matches!(
        hir.functions[2].instructions[0],
        Instruction::TryLet { .. }
    ));
    let Instruction::Let { value, .. } = &hir.functions[3].instructions[0] else {
        panic!("expected `?=` to retain the complete Result");
    };
    assert_eq!(value.ty(), Some(ValueType::Result));
}

#[test]
fn try_equal_rejects_non_result_expressions() {
    let ast = parse(&lex("def main():\n    value ?= 42\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error.message.contains("requires a fallible expression"));
}

#[test]
fn intrinsic_size_is_not_shadowed_by_a_linked_package_function() {
    let source = "def count(values: list[int]) -> int:\n    return size(values)\n";
    let interface = "def size(path: string) -> Result[int, IOError]:\n    return failure(IOError(\"unused\"))\n";
    let ast = parse(&lex(source).unwrap()).unwrap();
    let interface_ast = parse(&lex(interface).unwrap()).unwrap();
    let hir = analyze_with_interfaces(&ast, &[("file".into(), interface_ast)]).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected a return value");
    };
    assert_eq!(value.ty(), Some(ValueType::Int));
}

#[test]
fn rejects_unknown_functions() {
    let ast = parse(&lex("def main():\n    write(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "unknown function `write`");
}

#[test]
fn retains_typed_conditional_expressions_in_hir() {
    let source = "def reluValue(x: float) -> float:\n    return 0.0 if x < 0.0 else x\n";
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected a return value");
    };
    assert_eq!(value.ty(), Some(ValueType::Float));
    assert!(matches!(value.kind(), Expression::Conditional { .. }));
}

#[test]
fn retains_ranked_tensor_element_and_dimension_types_in_hir() {
    let source = concat!(
        "unsafe:\n    native(\"tensor_identity\") def identity(value: Tensor[f32, 2, dynamic]) -> Tensor[f32, 2, dynamic]\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let ValueType::Tensor(tensor) = hir.functions[0].params[0].ty else {
        panic!("expected tensor parameter");
    };
    assert_eq!(tensor.element, TensorElementType::F32);
    assert_eq!(tensor.rank, Some(2));
    assert_eq!(tensor.dimensions[0], TensorDimension::Static(2));
    assert_eq!(tensor.dimensions[1], TensorDimension::Dynamic);
    assert_eq!(hir.functions[0].return_type, ValueType::Tensor(tensor));
}

#[test]
fn rejects_incompatible_static_tensor_shapes_at_call_boundaries() {
    let source = concat!(
        "unsafe:\n    native(\"consume\") def consume(value: Tensor[f64, 2, 3])\n",
        "def wrong(value: Tensor[f64, 2, 4]):\n",
        "    consume(value)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error.message.contains("expected Tensor"));
}

#[test]
fn tensor_wildcard_parameters_infer_each_callers_dtype() {
    let source = concat!(
        "unsafe:\n",
        "    native(\"release_tensor\") def release[type](value: Tensor[type])\n",
        "\n",
        "def release_all(bfloat: Tensor[bf16], float: Tensor[f32], integer: Tensor[i64]):\n",
        "    release(bfloat)\n",
        "    release(float)\n",
        "    release(integer)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    assert_eq!(hir.functions[0].params[0].ty, ValueType::TensorAny);
}

#[test]
fn tensor_wildcard_parameters_reject_non_tensor_values() {
    let source = concat!(
        "unsafe:\n",
        "    native(\"release_tensor\") def release[type](value: Tensor[type])\n",
        "\n",
        "def wrong():\n",
        "    release(\"not a tensor\")\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error.message.contains("expected TensorAny"));
}

#[test]
fn retains_local_task_placement_after_validating_its_import() {
    let source = concat!(
        "import distributed\n",
        "\n",
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    with self and local:\n",
        "        task = async work()\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::With { instructions, .. } = &hir.main().unwrap().instructions[0] else {
        panic!("expected task context");
    };
    let Instruction::Let { value, .. } = &instructions[0] else {
        panic!("expected a task binding");
    };
    let Expression::Task { placement, .. } = value.kind() else {
        panic!("expected a task binding");
    };
    assert_eq!(*placement, TaskPlacement::Local);
}

#[test]
fn rejects_local_task_placement_without_the_distributed_import() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    task = async work() with self and local\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(
        error.message,
        "task placement `local` requires `import distributed`"
    );
}

#[test]
fn retains_parallel_placement_after_validating_the_import() {
    let source = concat!(
        "import parallel\n",
        "\n",
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    with self and simd:\n",
        "        task = async work()\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::With { instructions, .. } = &hir.main().unwrap().instructions[0] else {
        panic!("expected task context");
    };
    let Instruction::Let { value, .. } = &instructions[0] else {
        panic!("expected task binding");
    };
    let Expression::Task { placement, .. } = value.kind() else {
        panic!("expected task binding");
    };
    assert_eq!(*placement, TaskPlacement::Simd);
}

#[test]
fn retains_gpu_placement_on_a_parallel_for_region() {
    let source = concat!(
        "import parallel\n",
        "\n",
        "def main():\n",
        "    values := [1, 2]\n",
        "    for index in indices(values) with gpu:\n",
        "        values[index] += 1\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::With {
        placement,
        instructions,
        ..
    } = &hir.main().unwrap().instructions[1]
    else {
        panic!("expected placement region");
    };
    assert_eq!(*placement, TaskPlacement::Gpu);
    assert!(matches!(instructions[0], Instruction::For { .. }));
}

#[test]
fn rejects_gpu_regions_without_the_parallel_import() {
    let source = concat!(
        "def main():\n",
        "    values := [1]\n",
        "    with gpu:\n",
        "        values[0] += 1\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error
        .message
        .contains("execution placement `gpu` requires `import parallel`"));
}

#[test]
fn rejects_parallel_placement_without_the_parallel_import() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    task = async work() with self and gpu\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error
        .message
        .contains("task placement `gpu` requires `import parallel`"));
}

#[test]
fn accepts_snake_case_function_names_for_lint_managed_style() {
    let ast = parse(&lex("def bad_name():\n    print(\"hello\")\n").unwrap()).unwrap();
    analyze(&ast).unwrap();
}

#[test]
fn rejects_a_typed_function_that_does_not_return_on_every_path() {
    let source = "def choose(ready: bool) -> int:\n    if ready:\n        return 1\n";
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "function `choose` must return a value");
}

#[test]
fn type_checks_injected_return_values() {
    let source = concat!(
        "def read() -> int:\n",
        "    return 1\n",
        "\n",
        "test:\n",
        "    chaos.add(when read return \"not an int\")\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();

    let error = analyze(&ast).unwrap_err();

    assert_eq!(error.message, "E0202: expected Int, found String");
}

#[test]
fn type_checks_calls_against_imported_package_interfaces() {
    let interface =
        parse(&lex("def square(value: float) -> float:\n    return value * value\n").unwrap())
            .unwrap();
    let source = concat!(
        "import math\n",
        "\n",
        "def main():\n",
        "    square(\"three\")\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();

    let error = analyze_with_interfaces(&module, &[("math".into(), interface)]).unwrap_err();

    assert_eq!(error.message, "E0202: expected Float, found String");
}

#[test]
fn resolved_package_calls_retain_identity_and_signature() {
    let interface =
        parse(&lex("def square(value: float) -> float:\n    return value * value\n").unwrap())
            .unwrap();
    let module = parse(
        &lex(concat!(
            "import math\n",
            "\n",
            "def apply(value: float) -> float:\n",
            "    return math.square(value)\n",
        ))
        .unwrap(),
    )
    .unwrap();

    let hir = analyze_with_interfaces(&module, &[("math".into(), interface)]).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected package call return");
    };
    let Expression::Call { target, .. } = value.kind() else {
        panic!("expected resolved package call");
    };
    assert_eq!(target.name, "math.square");
    assert_eq!(
        target.id,
        severian_hir::FunctionId::from_name("math.square")
    );
    let signature = target.signature.as_ref().expect("resolved signature");
    assert_eq!(signature.parameters, [ValueType::Float]);
    assert_eq!(signature.returns, ValueType::Float);
}

#[test]
fn retains_formatted_string_operands_for_native_lowering() {
    let source = concat!(
        "def describe(label: string, value: float) -> string:\n",
        "    return f\"{label}: {value}\"\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected a formatted return value")
    };
    let Expression::Format {
        template,
        args,
        arg_types,
    } = value.kind()
    else {
        panic!("expected a formatted return value")
    };
    assert_eq!(template, "{label}: {value}");
    assert_eq!(
        args,
        &[
            Expression::Variable("label".into()),
            Expression::Variable("value".into()),
        ]
    );
    assert_eq!(arg_types, &[ValueType::String, ValueType::Float]);
}

#[test]
fn retains_first_class_function_return_types() {
    let source = concat!(
        "def apply(op: fn[int, int, int], left: int, right: int) -> int:\n",
        "    return op(left, right)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected an indirect function call")
    };
    let Expression::CallValue { return_type, .. } = value.kind() else {
        panic!("expected an indirect function call")
    };
    assert_eq!(*return_type, ValueType::Int);
}

#[test]
fn requires_classes_to_implement_their_declared_traits() {
    let missing = concat!(
        "trait Format:\n",
        "    kind() -> string\n",
        "\n",
        "class Broken: Format\n",
        "    path: string\n",
    );
    let ast = parse(&lex(missing).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error
        .message
        .contains("class `Broken` does not implement `kind` required by trait `Format`"));

    let incompatible = concat!(
        "trait Format:\n",
        "    kind() -> string\n",
        "\n",
        "class Broken: Format\n",
        "    def kind() -> int:\n",
        "        return 1\n",
    );
    let ast = parse(&lex(incompatible).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error.message.contains("does not match trait `Format`"));
    assert!(error.message.contains("expected `kind() -> string`"));
}

#[test]
fn validates_traits_imported_from_packages() {
    let interface_source = "trait File:\n    kind() -> string\n";
    let interface = parse(&lex(interface_source).unwrap()).unwrap();
    let source = concat!(
        "import file\n",
        "\n",
        "class Playlist: file.File\n",
        "    def kind() -> string:\n",
        "        return \"playlist\"\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    analyze_with_interfaces(&ast, &[("file".into(), interface)]).unwrap();
}

#[test]
fn refines_literal_file_reads_to_the_extension_class() {
    let interface_source = concat!(
        "trait File:\n",
        "    kind() -> string\n",
        "\n",
        "class WAV: File\n",
        "    sample_rate: int\n",
        "\n",
        "    def kind() -> string:\n",
        "        return \"wav\"\n",
        "\n",
        "def read(path: string) -> Result[File, string]:\n",
        "    return failure(\"unavailable\")\n",
    );
    let interface = parse(&lex(interface_source).unwrap()).unwrap();

    let literal_source = concat!(
        "import file\n",
        "\n",
        "def sample_rate() -> int:\n",
        "    audio = file.read(\"sound.WAV\")\n",
        "    return audio.sample_rate\n",
        "\n",
        "def switched_sample_rate() -> int:\n",
        "    switch file.read(\"sound.wav\"):\n",
        "        ok audio:\n",
        "            return audio.sample_rate\n",
        "        failure error:\n",
        "            return 0\n",
    );
    let literal = parse(&lex(literal_source).unwrap()).unwrap();
    let literal_hir =
        analyze_with_interfaces(&literal, &[("file".into(), interface.clone())]).unwrap();
    let Instruction::TryLet {
        receiver: Some(receiver),
        ..
    } = &literal_hir.functions[0].instructions[0]
    else {
        panic!("expected fallible assignment to preserve its success receiver");
    };
    assert_eq!(receiver.name, "WAV");

    let dynamic_source = concat!(
        "import file\n",
        "\n",
        "def sample_rate(path: string) -> int:\n",
        "    audio = file.read(path)\n",
        "    return audio.sample_rate\n",
    );
    let dynamic = parse(&lex(dynamic_source).unwrap()).unwrap();
    let error = analyze_with_interfaces(&dynamic, &[("file".into(), interface)]).unwrap_err();
    assert!(error
        .message
        .contains("class `File` has no field `sample_rate`"));
}

#[test]
fn attaches_source_and_structural_type_metadata_without_changing_legacy_hir() {
    let source = concat!(
        "enum Outcome:\n",
        "    Found(value: int)\n",
        "    Missing\n",
        "\n",
        "class Box:\n",
        "    values: list[int]\n",
        "\n",
        "unsafe:\n    native(\"load_values\") def loadValues(values: list[int]) -> Result[list[int], string]\n",
        "\n",
        "def choose() -> Outcome:\n",
        "    return Found(1)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let mut hir = analyze(&ast).unwrap();
    attach_module_metadata(&ast, &mut hir, "/workspace/model.sev", source, None);

    let files = hir.metadata.sources.files();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path.to_string_lossy(), "/workspace/model.sev");
    assert_eq!(files[0].source, source);

    let load = hir.metadata.functions[&FunctionId::from_name("loadValues")].clone();
    let TypeKind::List(element) = hir.metadata.types.get(load.parameters[0]).unwrap() else {
        panic!("expected a detailed list parameter")
    };
    assert_eq!(hir.metadata.types.get(*element), Some(&TypeKind::Int));
    let TypeKind::Result { ok, error } = hir.metadata.types.get(load.returns).unwrap() else {
        panic!("expected a detailed result return")
    };
    assert!(matches!(
        hir.metadata.types.get(*ok),
        Some(TypeKind::List(_))
    ));
    assert_eq!(hir.metadata.types.get(*error), Some(&TypeKind::String));

    let box_id = TypeDefinitionId::from_name("Box");
    let box_definition = &hir.metadata.classes[&box_id];
    assert_eq!(box_definition.fields[0].name, "values");
    assert!(matches!(
        hir.metadata.types.get(box_definition.fields[0].ty),
        Some(TypeKind::List(_))
    ));

    let outcome_id = TypeDefinitionId::from_name("Outcome");
    let found_id = VariantId::from_name("Found");
    assert_eq!(hir.metadata.enums[&outcome_id].variants[0].id, found_id);
    assert!(hir
        .metadata
        .sources
        .definition_span(DefinitionId::Variant {
            owner: outcome_id,
            variant: found_id,
        })
        .is_some());

    let Instruction::Return(Some(value)) = &hir
        .functions
        .iter()
        .find(|function| function.name == "choose")
        .unwrap()
        .instructions[0]
    else {
        panic!("expected a return value")
    };
    assert!(hir
        .metadata
        .sources
        .expression_span(value.hir_id().unwrap())
        .is_some());
    let Expression::Variant { type_id, .. } = value.kind() else {
        panic!("expected an enum variant")
    };
    assert_eq!(*type_id, Some(outcome_id));
}

#[test]
fn imported_function_returns_keep_the_canonical_class_definition() {
    let data_source = concat!(
        "class Data:\n",
        "    def filter() -> Data:\n",
        "        return Data()\n",
    );
    let data_module = parse(&lex(data_source).unwrap()).unwrap();
    let interface = PackageInterface {
        name: "data".into(),
        export_package: None,
        module: data_module,
        compiler: Default::default(),
        source_path: PathBuf::from("/workspace/data.sev"),
        source: data_source.into(),
    };
    let helper_source = concat!(
        "import data as tabular\n",
        "\n",
        "def make() -> tabular.Data:\n",
        "    return tabular.Data()\n",
    );
    let helper = parse(&lex(helper_source).unwrap()).unwrap();
    let mut hir = analyze_with_packages(&helper, std::slice::from_ref(&interface)).unwrap();
    attach_module_metadata_with_packages(
        &helper,
        &mut hir,
        "/workspace/helper.sev",
        helper_source,
        Some("helper"),
        &[interface],
    );

    let signature = &hir.metadata.functions[&FunctionId::from_name("helper.make")];
    let TypeKind::Named {
        definition, name, ..
    } = hir.metadata.types.get(signature.returns).unwrap()
    else {
        panic!("expected imported return type to retain its nominal class")
    };
    assert_eq!(*definition, TypeDefinitionId::from_name("data.Data"));
    assert_eq!(name, "tabular.Data");
}

#[test]
fn ordinary_unannotated_parameters_default_to_any() {
    let ast = parse(&lex("def identity(value) -> Any:\n    return value\n").unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    assert_eq!(hir.functions[0].params[0].ty, ValueType::Any);
    assert_eq!(hir.functions[0].return_type, ValueType::Any);
}

#[test]
fn dynamically_gets_and_sets_known_object_fields_without_losing_field_types() {
    let source = concat!(
        "class Point:\n",
        "    x: int\n",
        "    y: int\n",
        "\n",
        "def update() -> int:\n",
        "    point := Point(3, 4)\n",
        "    point.set(\"x\", 10)\n",
        "    return point.get(\"x\")\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let function = hir
        .functions
        .iter()
        .find(|function| function.name == "update")
        .unwrap();
    let Instruction::Return(Some(value)) = &function.instructions[2] else {
        panic!("expected the dynamically selected field")
    };
    assert_eq!(value.ty(), Some(ValueType::Int));
}

#[test]
fn dynamic_object_set_obeys_mutability_and_known_field_types() {
    let immutable = concat!(
        "class Point:\n",
        "    x: int\n",
        "\n",
        "def invalid():\n",
        "    point = Point(3)\n",
        "    point.set(\"x\", 10)\n",
    );
    let ast = parse(&lex(immutable).unwrap()).unwrap();
    assert!(analyze(&ast)
        .unwrap_err()
        .message
        .contains("is not changeable"));

    let wrong_type = concat!(
        "class Point:\n",
        "    x: int\n",
        "\n",
        "def invalid():\n",
        "    point := Point(3)\n",
        "    point.set(\"x\", \"wrong\")\n",
    );
    let ast = parse(&lex(wrong_type).unwrap()).unwrap();
    assert!(analyze(&ast).unwrap_err().message.contains("E0202"));

    let missing_field = concat!(
        "class Point:\n",
        "    x: int\n",
        "\n",
        "def invalid():\n",
        "    point = Point(3)\n",
        "    print(point.get(\"z\"))\n",
    );
    let ast = parse(&lex(missing_field).unwrap()).unwrap();
    assert!(analyze(&ast)
        .unwrap_err()
        .message
        .contains("has no field `z`"));
}
