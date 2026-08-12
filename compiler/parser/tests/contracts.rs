use severian_ast::{Item, TestMode};
use severian_lexer::lex;
use severian_parser::parse;

#[test]
fn parses_function_and_profile_test_contracts() {
    let source = concat!(
        "def bounded(x: int) -> int with\n",
        "{\n",
        "    x >= 0,\n",
        "    defer x < 10 -> exception(\"x is too large\", location, vars),\n",
        "}:\n",
        "    return x\n",
        "test with profile \"bounds\" -> TestResult with\n",
        "{\n",
        "    defer 1ms < time < 2ms -> exception(\"bad runtime\", location, vars),\n",
        "    defer memory < 32mb -> exception(\"too much memory\"),\n",
        "}:\n",
        "    assert(bounded(1) == 1)\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let contract = function.contract.as_ref().unwrap();
    assert_eq!(contract.clauses.len(), 2);
    assert!(!contract.clauses[0].deferred);
    assert!(contract.clauses[1].deferred);
    let failure = contract.clauses[1].failure.as_ref().unwrap();
    assert!(failure.location);
    assert!(failure.vars);

    let test = &function.tests[0];
    assert!(test.modes.contains(&TestMode::Profile));
    assert!(test.return_type.is_some());
    assert_eq!(test.contract.as_ref().unwrap().clauses.len(), 2);
}

#[test]
fn requires_a_trailing_comma_on_every_contract_clause() {
    let source = "def invalid(x: int) -> int with\n{\n    x >= 0\n}:\n    return x\n";
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert!(error.message.contains("`,` after every contract clause"));
}

#[test]
fn rejects_unknown_contract_exception_options() {
    let source = concat!(
        "def invalid(x: int) -> int with\n",
        "{\n",
        "    x >= 0 -> exception(\"invalid\", context),\n",
        "}:\n",
        "    return x\n",
    );
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert!(error.message.contains("`location` and `vars`"));
}

#[test]
fn requires_with_before_function_and_test_contracts() {
    for source in [
        "def invalid(x: int) -> int\n{\n    x >= 0,\n}:\n    return x\n",
        "def support():\n    return\ntest \"invalid\"\n{\n    true,\n}:\n    support()\n",
    ] {
        let error = parse(&lex(source).unwrap()).unwrap_err();
        assert!(error.message.contains("contracts require `with`"));
    }
}
