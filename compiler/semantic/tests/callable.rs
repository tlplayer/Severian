use severian_hir::{Expression, Instruction, ValueType};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

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
