use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

fn analyze_error(source: &str) -> String {
    let module = parse(&lex(source).unwrap()).unwrap();
    analyze(&module).unwrap_err().message
}

#[test]
fn suggests_an_unambiguous_named_argument_typo() {
    let message = analyze_error(
        "def load(name: string, device: string):\n    return\n\ndef main():\n    load(\"Qwen\", devcie = \"cuda\")\n",
    );
    assert_eq!(
        message,
        "E000204: unknown argument `devcie`; did you mean `device`?"
    );
}

#[test]
fn enum_switches_must_cover_every_variant() {
    let message = analyze_error(
        "enum Status:\n    Ready\n    Failed\n    Waiting\n\ndef run(status: Status):\n    switch status:\n        Ready:\n            return\n        Failed:\n            return\n",
    );
    assert_eq!(
        message,
        "E000206: non-exhaustive switch on `Status`; missing `Status.Waiting`"
    );
}

#[test]
fn exhaustive_enum_switches_remain_valid() {
    let source = "enum Status:\n    Ready\n    Failed\n\ndef run(status: Status):\n    switch status:\n        Ready:\n            return\n        Failed:\n            return\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    analyze(&module).unwrap();
}

#[test]
fn constant_zero_divisors_fail_before_lowering() {
    let message = analyze_error("def main():\n    print(100 / 0)\n");
    assert_eq!(
        message,
        "E000502: division by zero is known at compile time"
    );
}

#[test]
fn compiler_function_names_cannot_be_redeclared_at_module_scope() {
    for name in ["size", "len", "sqrt", "min", "max"] {
        let source = format!("def {name}(value: int) -> int:\n    return value\n");
        let ast = parse(&lex(&source).unwrap()).unwrap();
        let error = analyze(&ast).unwrap_err();
        assert_eq!(
            error.message,
            format!("E000208: `{name}` is reserved for a compiler-provided function")
        );
    }
}

#[test]
fn compiler_function_names_remain_legal_in_method_namespaces() {
    let source = concat!(
        "class Buffer:\n",
        "    values: list[int]\n",
        "\n",
        "    def size() -> int:\n",
        "        return len(values)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    analyze(&ast).unwrap();
}

#[test]
fn explicit_self_parameters_are_rejected_before_generic_specialization() {
    let message = analyze_error(concat!(
        "class Accumulator[T]:\n",
        "    value: T\n",
        "\n",
        "    def current(self) -> T:\n",
        "        return value\n",
    ));
    assert_eq!(
        message,
        "E000209: `self` is an implicit class receiver and must not be declared as a parameter"
    );
}

#[test]
fn nested_functions_cannot_claim_the_implicit_receiver_name() {
    let message = analyze_error(concat!(
        "def outer():\n",
        "    def inner(self):\n",
        "        return\n",
    ));
    assert_eq!(
        message,
        "E000209: `self` is an implicit class receiver and must not be declared as a parameter"
    );
}

#[test]
fn implicit_receivers_keep_fields_and_self_available() {
    let source = concat!(
        "class Accumulator:\n",
        "    value: int\n",
        "\n",
        "    def current() -> int:\n",
        "        return value\n",
        "\n",
        "    def receiver() -> Accumulator:\n",
        "        return self\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    analyze(&ast).unwrap();
}

#[test]
fn recoverable_results_cannot_leak_into_binary_operators() {
    let message = analyze_error(concat!(
        "def checked() -> Result[int, string]:\n",
        "    return 3\n",
        "\n",
        "def main():\n",
        "    assert(checked() == 3)\n",
    ));
    assert_eq!(
        message,
        "E000801: a recoverable Result cannot be used as an operator operand; bind it to propagate success or handle it with switch"
    );
}
