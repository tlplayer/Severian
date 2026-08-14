use severian_hir::{Expression, Instruction, ValueType};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::{analyze, attach_module_metadata};
use std::path::PathBuf;

#[test]
fn ordinary_unannotated_parameters_default_to_any() {
    let ast = parse(&lex("def identity(value) -> Any:\n    return value\n").unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    assert_eq!(hir.functions[0].params[0].ty, ValueType::Any);
    assert_eq!(hir.functions[0].return_type, ValueType::Any);
}

#[test]
fn overloaded_constructors_receive_distinct_stable_identities() {
    let source = concat!(
        "class Point:\n",
        "    value: int\n",
        "    def Point(x: int):\n        value = x\n",
        "    def Point(x: int, y: int):\n        value = x + y\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let mut hir = analyze(&ast).unwrap();
    attach_module_metadata(
        &ast,
        &mut hir,
        PathBuf::from("overloaded.sev"),
        source,
        None,
    );

    let constructors = &hir.classes[0].constructors;
    assert_ne!(constructors[0].id, constructors[1].id);
    assert!(hir.metadata.functions.contains_key(&constructors[0].id));
    assert!(hir.metadata.functions.contains_key(&constructors[1].id));
}

#[test]
fn resolved_bindings_distinguish_shadowed_names() {
    let source = concat!(
        "def choose(value: int) -> int:\n",
        "    callback = |value| value\n",
        "    return value\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let function = &hir.functions[0];
    let outer = &function.params[0].name;
    let Instruction::Let { value, .. } = &function.instructions[0] else {
        panic!("expected callback binding");
    };
    let Expression::Lambda { params, body } = value.kind() else {
        panic!("expected lambda expression");
    };
    let Expression::Variable(lambda_use) = body.kind() else {
        panic!("expected lambda parameter use");
    };
    let Instruction::Return(Some(returned)) = &function.instructions[1] else {
        panic!("expected return expression");
    };
    let Expression::Variable(outer_use) = returned.kind() else {
        panic!("expected outer parameter use");
    };

    assert_eq!(params[0].name, outer.name);
    assert_ne!(params[0].id, outer.id);
    assert_eq!(lambda_use.id, params[0].id);
    assert_eq!(outer_use.id, outer.id);
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
