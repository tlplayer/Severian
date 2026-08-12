use severian_ast::{Expr, Item, Literal, Stmt, TaskOwner, TaskPlacement};
use severian_lexer::lex;
use severian_parser::parse;

#[test]
fn parses_the_hello_fixture_into_the_source_ast() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let tokens = lex(source).unwrap();
    let module = parse(&tokens).unwrap();

    let Item::Function(main) = &module.items[0] else {
        panic!("expected a function");
    };
    assert_eq!(main.name.name, "main");

    let Stmt::Expr(Expr::Call(call)) = &main.body.statements[0] else {
        panic!("expected a call statement");
    };
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        panic!("expected an identifier callee");
    };
    assert_eq!(callee.name, "print");
    assert!(matches!(
        &call.args[0].value,
        Expr::Literal(Literal::String { value, .. }) if value == "hello, severian"
    ));
}

#[test]
fn rejects_a_missing_function_body() {
    let tokens = lex("def main():\n").unwrap();
    let error = parse(&tokens).unwrap_err();
    assert!(error.message.contains("indented function body"));
}

#[test]
fn parses_negated_membership_as_a_comparison() {
    let source = concat!(
        "def absent(value: int, values: list[int]) -> bool:\n",
        "    return value not in values\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::Return(statement) = &function.body.statements[0] else {
        panic!("expected return");
    };
    assert!(matches!(statement.value.as_ref(), Some(Expr::Unary(_))));
}

#[test]
fn parses_a_for_loop_setup_binding() {
    let source = concat!(
        "def sum(values: list[int]) -> int:\n",
        "    for value in values with total := 0:\n",
        "        total += value\n",
        "    return total\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::For(statement) = &function.body.statements[0] else {
        panic!("expected for loop");
    };
    assert!(matches!(statement.setup.as_deref(), Some(Stmt::Let(_))));
}

#[test]
fn parses_an_explicit_native_abi_declaration() {
    let source = concat!(
        "unsafe:\n",
        "    native(\"__sev_file_read\") def fileRead(\n",
        "        path: string,\n",
        "    ) -> Result[string, IOError]\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected native function declaration");
    };

    assert_eq!(function.name.name, "fileRead");
    assert_eq!(function.native_symbol.as_deref(), Some("__sev_file_read"));
    assert!(function.body.statements.is_empty());
}

#[test]
fn permits_dynamic_parameters_in_ordinary_functions() {
    let module = parse(&lex("def identity(value):\n    return value\n").unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    assert!(function.params[0].ty.is_none());
}

#[test]
fn permits_dynamic_fields_when_they_have_defaults() {
    let source = "class Box:\n    value = 1\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Class(class) = &module.items[0] else {
        panic!("expected class");
    };
    assert!(class.fields[0].ty.is_none());
    assert!(class.fields[0].default.is_some());
}

#[test]
fn native_abi_parameters_must_remain_explicitly_typed() {
    let source = "unsafe:\n    native(\"host_value\") def hostValue(value) -> int\n";
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "native ABI parameters require explicit types"
    );
}

#[test]
fn rejects_a_native_abi_declaration_without_unsafe() {
    let error = parse(&lex("native(\"host_call\") def hostCall()\n").unwrap()).unwrap_err();

    assert_eq!(
        error.message,
        "native declarations cross the host ABI and require an `unsafe:` block"
    );
}

#[test]
fn rejects_an_inline_unsafe_native_declaration() {
    let error = parse(&lex("unsafe native(\"host_call\") def hostCall()\n").unwrap()).unwrap_err();

    assert_eq!(error.message, "expected `:` after unsafe");
}

#[test]
fn parses_local_task_placement_in_the_with_clause() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    with self and local:\n",
        "        task = async work()\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(main) = &module.items[1] else {
        panic!("expected main function");
    };
    let Stmt::With(block) = &main.body.statements[0] else {
        panic!("expected task context");
    };
    let Stmt::Let(binding) = &block.body.statements[0] else {
        panic!("expected task binding");
    };
    let Expr::Async(task) = binding.value.as_ref().unwrap() else {
        panic!("expected async expression");
    };
    assert_eq!(task.owner, TaskOwner::SelfOwned);
    assert_eq!(task.placement, TaskPlacement::Local);
    assert!(task.captures.is_empty());
}

#[test]
fn parses_parallel_placement_in_the_with_clause() {
    let source = concat!(
        "import parallel\n",
        "\n",
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    with self and gpu:\n",
        "        task = async work()\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(main) = &module.items[2] else {
        panic!("expected main function");
    };
    let Stmt::With(block) = &main.body.statements[0] else {
        panic!("expected with block");
    };
    let Stmt::Let(binding) = &block.body.statements[0] else {
        panic!("expected task binding");
    };
    let Some(Expr::Async(task)) = &binding.value else {
        panic!("expected async expression");
    };
    assert_eq!(task.placement, TaskPlacement::Gpu);
}

#[test]
fn parses_gpu_suffix_on_a_for_loop_as_an_execution_region() {
    let source = concat!(
        "def main():\n",
        "    values := [1, 2]\n",
        "    for index in indices(values) with gpu:\n",
        "        values[index] += 1\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(main) = &module.items[0] else {
        panic!("expected main function");
    };
    let Stmt::With(region) = &main.body.statements[1] else {
        panic!("expected placement region");
    };
    assert!(matches!(&region.resources[0], Expr::Identifier(name) if name.name == "gpu"));
    assert!(matches!(&region.body.statements[0], Stmt::For(_)));
}

#[test]
fn parses_one_off_gpu_placement_without_an_explicit_owner() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    task = async work() with gpu\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(main) = &module.items[1] else {
        panic!("expected main function");
    };
    let Stmt::Let(binding) = &main.body.statements[0] else {
        panic!("expected task binding");
    };
    let Some(Expr::Async(task)) = &binding.value else {
        panic!("expected async expression");
    };
    assert_eq!(task.owner, TaskOwner::SelfOwned);
    assert_eq!(task.placement, TaskPlacement::Gpu);
}

#[test]
fn rejects_user_directed_kernel_fusion() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    with self and fuse:\n",
        "        task = async work()\n",
    );
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert!(error.message.contains("kernel fusion is automatic"));
}

#[test]
fn rejects_a_bare_async_expression_without_an_enclosing_task_context() {
    let source = concat!(
        "def work() -> int:\n",
        "    return 1\n",
        "\n",
        "def main():\n",
        "    task = async work()\n",
    );
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert!(error.message.contains("outside a task context"));
}

#[test]
fn accepts_chaos_injection_patterns_in_any_test_block() {
    let source = concat!(
        "def read():\n",
        "    return None\n",
        "\n",
        "test:\n",
        "    chaos.add(when read return None)\n",
        "\n",
        "test with chaos \"throws\":\n",
        "    chaos.add(when read throw TimedOut)\n",
    );

    parse(&lex(source).unwrap()).unwrap();
}

#[test]
fn parses_integration_tests_with_output_assertions() {
    let source = concat!(
        "def main():\n",
        "    print(\"hello\")\n",
        "\n",
        "test with integration \"native output\":\n",
        "    main()\n",
        "    assert(\"hello\" in stdout)\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    severian_parser::parse(&tokens).unwrap();
}

#[test]
fn parses_else_condition_as_an_ordinary_conditional_branch() {
    let source = concat!(
        "def classify(value: int) -> int:\n",
        "    if value > 0:\n",
        "        return 1\n",
        "    else value < 0:\n",
        "        return -1\n",
        "    else:\n",
        "        return 0\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::If(statement) = &function.body.statements[0] else {
        panic!("expected conditional");
    };
    assert!(matches!(
        statement.else_branch,
        Some(severian_ast::ElseBranch::If(_))
    ));
}

#[test]
fn keeps_elif_as_a_compatibility_spelling() {
    let source = concat!(
        "def classify(value: int) -> int:\n",
        "    if value > 0:\n",
        "        return 1\n",
        "    elif value < 0:\n",
        "        return -1\n",
        "    else:\n",
        "        return 0\n",
    );
    parse(&lex(source).unwrap()).unwrap();
}

#[test]
fn parses_imported_decorator_symbol_packs() {
    let source = concat!(
        "import math\n",
        "\n",
        "@math(X, *, ^)\n",
        "def transform(value: int) -> int:\n",
        "    return value\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let severian_ast::Item::Function(function) = &module.items[1] else {
        panic!("expected decorated function");
    };
    assert_eq!(function.decorators[0].name.segments[0].name, "math");
    assert_eq!(
        function.decorators[0]
            .symbols
            .iter()
            .map(|symbol| symbol.spelling.as_str())
            .collect::<Vec<_>>(),
        ["X", "*", "^"]
    );
}

#[test]
fn parses_bare_decorators_without_arguments() {
    let source = concat!(
        "import tensor\n",
        "\n",
        "@tensor\n",
        "def transform(value: int) -> int:\n",
        "    return value\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[1] else {
        panic!("expected decorated function");
    };
    assert_eq!(function.decorators[0].name.segments[0].name, "tensor");
    assert!(function.decorators[0].symbols.is_empty());
}

#[test]
fn rejects_empty_decorator_parentheses() {
    let source = "import tensor\n\n@tensor()\ndef transform():\n    return\n";
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert!(error.message.contains("write the decorator without `()`"));
}

#[test]
fn parses_quoted_local_imports_with_optional_aliases() {
    let module =
        parse(&lex("import \"helpers.sev\"\nimport \"local/math\" as m\n").unwrap()).unwrap();
    let Item::Import(first) = &module.items[0] else {
        panic!("expected local import");
    };
    assert!(matches!(
        &first.kind,
        severian_ast::ImportKind::Local { path, alias: None } if path == "helpers.sev"
    ));
    let Item::Import(second) = &module.items[1] else {
        panic!("expected aliased local import");
    };
    assert!(matches!(
        &second.kind,
        severian_ast::ImportKind::Local { path, alias: Some(alias) }
            if path == "local/math" && alias.name == "m"
    ));
}

#[test]
fn parses_piecewise_conditional_expressions() {
    let source = "def reluValue(x: float) -> float:\n    return 0.0 if x < 0.0 else x\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::Return(statement) = &function.body.statements[0] else {
        panic!("expected return");
    };
    assert!(matches!(statement.value, Some(Expr::If(_))));
}

#[test]
fn parses_server_signatures_destructuring_resources_and_contracts() {
    let source = concat!(
        "import network\n",
        "\n",
        "def serve(\n",
        "    connection: network.TCPConnection,\n",
        ") -> Result[unit, IOError] with {\n",
        "    connection != invalid,\n",
        "    with connection,\n",
        "}:\n",
        "    reader, writer = connection.split()\n",
        "    with connection:\n",
        "        while true with connection:\n",
        "            return\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let severian_ast::Item::Function(function) = &module.items[1] else {
        panic!("expected function");
    };
    assert_eq!(
        function.params[0].ty.as_ref().unwrap().span().start,
        source.find("network.TCPConnection").unwrap()
    );
    assert!(function.contract.is_some());
}

#[test]
fn rejects_chaos_injection_patterns_outside_tests() {
    let source = concat!(
        "def read():\n",
        "    return None\n",
        "\n",
        "def main():\n",
        "    chaos.add(when read return None)\n",
    );

    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "chaos injection patterns are only valid inside tests"
    );
}

#[test]
fn restricts_address_of_to_unsafe_blocks() {
    let valid = concat!(
        "def first(values: list[int]) -> int:\n",
        "    unsafe:\n",
        "        pointer = &values\n",
        "        return pointer[0]\n",
    );
    parse(&lex(valid).unwrap()).unwrap();

    let invalid = concat!(
        "def first(values: list[int]) -> int:\n",
        "    pointer = &values\n",
        "    return pointer[0]\n",
    );
    let error = parse(&lex(invalid).unwrap()).unwrap_err();
    assert_eq!(error.message, "address-of is only valid inside `unsafe`");
}

#[test]
fn rejects_unsafe_blocks_inside_tests() {
    let source = concat!(
        "def probe():\n",
        "    return\n",
        "\n",
        "test \"unsafe is never a test capability\":\n",
        "    unsafe:\n",
        "        value = 1\n",
    );
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(error.message, "tests may not contain `unsafe` blocks");
}
