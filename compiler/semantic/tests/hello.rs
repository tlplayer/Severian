use severian_hir::{
    DefinitionId, Expression, FunctionId, Instruction, TaskPlacement, TensorDimension,
    TensorElementType, TypeDefinitionId, TypeKind, ValueType, VariantId,
};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::{analyze, analyze_with_interfaces, attach_module_metadata};

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
fn ordinary_unannotated_parameters_default_to_any() {
    let ast = parse(&lex("def identity(value) -> Any:\n    return value\n").unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    assert_eq!(hir.functions[0].params[0].ty, ValueType::Any);
    assert_eq!(hir.functions[0].return_type, ValueType::Any);
}
