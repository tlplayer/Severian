use severian_hir::{Expression, Instruction, ValueType};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::{analyze, analyze_with_interfaces};

#[test]
fn resolves_print_and_lowers_hello_to_hir() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    assert_eq!(
        hir.main().unwrap().instructions,
        vec![Instruction::Print(Expression::String(
            "hello, severian".into()
        ))]
    );
}

#[test]
fn rejects_unknown_functions() {
    let ast = parse(&lex("def main():\n    write(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "unknown function `write`");
}

#[test]
fn rejects_snake_case_function_names() {
    let ast = parse(&lex("def bad_name():\n    print(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "function `bad_name` must use lowerCamelCase");
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

    assert_eq!(error.message, "expected Int, found String");
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

    assert_eq!(error.message, "expected Float, found String");
}

#[test]
fn retains_formatted_string_operands_for_native_lowering() {
    let source = concat!(
        "def describe(label: string, value: float) -> string:\n",
        "    return f\"{label}: {value}\"\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    let Instruction::Return(Some(Expression::Format {
        template,
        args,
        arg_types,
    })) = &hir.functions[0].instructions[0]
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

    let Instruction::Return(Some(Expression::CallValue { return_type, .. })) =
        &hir.functions[0].instructions[0]
    else {
        panic!("expected an indirect function call")
    };
    assert_eq!(*return_type, ValueType::Int);
}
