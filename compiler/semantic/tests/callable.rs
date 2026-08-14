use severian_hir::{Expression, Instruction, ValueType};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

#[test]
fn declared_functions_shadow_contextual_variant_constructors() {
    let source = concat!(
        "def present(value: int) -> bool:\n",
        "    return value > 0\n",
        "\n",
        "def failure(value: int) -> bool:\n",
        "    return value < 0\n",
        "\n",
        "def main():\n",
        "    assert(present(1))\n",
        "    assert(failure(-1))\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    analyze(&module).unwrap();
}

#[test]
fn class_instances_with_forward_are_callable() {
    let source = concat!(
        "class Network:\n",
        "    def forward(value: int) -> int:\n",
        "        return value\n",
        "\n",
        "def predict(network: Network) -> int:\n",
        "    return network(42)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let Instruction::Return(Some(value)) = &hir.functions[0].instructions[0] else {
        panic!("expected a callable forward return");
    };
    assert!(matches!(
        value.kind(),
        Expression::MethodCall { method, .. } if method == "forward"
    ));
    assert_eq!(value.ty(), Some(ValueType::Int));
}

#[test]
fn class_instances_without_forward_explain_the_callable_contract() {
    let source = concat!(
        "class Dataset:\n",
        "    size: int\n",
        "\n",
        "def invalid(dataset: Dataset):\n",
        "    dataset(42)\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert!(error
        .message
        .contains("define `forward` to make it callable"));
}

#[test]
fn class_methods_can_return_the_implicit_receiver() {
    let source = concat!(
        "class Counter:\n",
        "    value: int\n",
        "\n",
        "    def increment(self) -> Counter:\n",
        "        value += 1\n",
        "        return self\n",
        "\n",
        "def incremented(counter: Counter) -> Counter:\n",
        "    return counter.increment()\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();
    let method = &hir.classes[0].methods[0];
    let Instruction::Return(Some(value)) = &method.instructions[1] else {
        panic!("expected the method to return its receiver");
    };
    assert!(matches!(
        value.kind(),
        Expression::Variable(binding) if binding.name == "self"
    ));
    assert!(
        method.params.is_empty(),
        "self is an implicit ABI parameter"
    );
}
