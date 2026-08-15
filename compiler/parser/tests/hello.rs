use severian_ast::{BinaryOp, Expr, Item, LetKind, Literal, Stmt, TaskOwner, TaskPlacement};
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
fn preserves_typed_changeable_bindings() {
    let source = "def main():\n    value: Any := 1\n    value = \"updated\"\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(main) = &module.items[0] else {
        panic!("expected a function");
    };
    let Stmt::Let(binding) = &main.body.statements[0] else {
        panic!("expected a typed binding");
    };
    assert_eq!(binding.kind, LetKind::Changeable);
    assert!(binding.ty.is_some());
}

#[test]
fn preserves_generic_parameters_and_capability_constraints() {
    let source = "def add[T: Numeric + Float](left: Tensor[T], right: Tensor[T]) -> Tensor[T]:\n    return left\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    assert_eq!(function.generic_params.len(), 1);
    assert_eq!(function.generic_params[0].name.name, "T");
    assert_eq!(function.generic_params[0].constraints.len(), 2);
}

#[test]
fn preserves_bounded_generic_classes_and_traits() {
    let source = concat!(
        "trait Module[T: TensorDType]:\n",
        "    def forward(x: Tensor[T]) -> Tensor[T]\n",
        "\n",
        "class Linear[T: TensorDType + Serializable]:\n",
        "    weight: Tensor[T]\n",
        "\n",
        "    def forward(x: Tensor[T]) -> Tensor[T]:\n",
        "        return x\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();

    let Item::Trait(module_trait) = &module.items[0] else {
        panic!("expected generic trait");
    };
    assert_eq!(module_trait.generic_params.len(), 1);
    assert_eq!(module_trait.generic_params[0].name.name, "T");
    assert_eq!(module_trait.generic_params[0].constraints.len(), 1);
    assert_eq!(module_trait.methods[0].params[0].name.name, "x");

    let Item::Class(linear) = &module.items[1] else {
        panic!("expected generic class");
    };
    assert_eq!(linear.generic_params.len(), 1);
    assert_eq!(linear.generic_params[0].name.name, "T");
    assert_eq!(linear.generic_params[0].constraints.len(), 2);
    assert_eq!(linear.methods[0].params[0].name.name, "x");
}

#[test]
fn parses_composed_traits_and_operator_contracts_without_inheritance_keywords() {
    let source = concat!(
        "trait Bits[T]:\n",
        "    operator |(a: T, b: T) -> T\n",
        "    operator &(a: T, b: T) -> T\n",
        "    operator ^(a: T, b: T) -> T\n",
        "\n",
        "trait Flags[T]:\n",
        "    Bits[T]\n",
        "    def enabled(flag: T) -> bool\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Trait(bits) = &module.items[0] else {
        panic!("expected Bits trait");
    };
    assert!(bits.composed_traits.is_empty());
    assert_eq!(
        bits.operators
            .iter()
            .map(|operator| operator.symbol.as_str())
            .collect::<Vec<_>>(),
        ["|", "&", "^"]
    );
    let Item::Trait(flags) = &module.items[1] else {
        panic!("expected Flags trait");
    };
    assert_eq!(flags.composed_traits.len(), 1);
    assert_eq!(flags.methods[0].name.name, "enabled");
}

#[test]
fn parses_provenance_aware_trait_headers_decorators_and_at_operator() {
    let source = concat!(
        "trait XLA:\n",
        "    @xla\n",
        "    operator @(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]\n",
        "\n",
        "trait Triton:\n",
        "    @triton\n",
        "    operator @(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]\n",
        "\n",
        "trait Tensor: XLA + Triton\n",
        "    @tensor(\n",
        "        backend = auto,\n",
        "        device = auto,\n",
        "    )\n",
        "\n",
        "@tensor(xla)\n",
        "def multiply(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]:\n",
        "    return left @ right\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Trait(tensor) = &module.items[2] else {
        panic!("expected Tensor trait");
    };
    assert_eq!(tensor.composed_traits.len(), 2);
    assert_eq!(tensor.decorators[0].name.segments[0].name, "tensor");
    assert_eq!(tensor.decorators[0].symbols[0].spelling, "backend");
    assert_eq!(
        tensor.decorators[0].symbols[0].value.as_deref(),
        Some("auto")
    );
    let Item::Function(multiply) = &module.items[3] else {
        panic!("expected multiply function");
    };
    let Stmt::Return(return_) = &multiply.body.statements[0] else {
        panic!("expected return");
    };
    let Some(Expr::Binary(operation)) = &return_.value else {
        panic!("expected matrix multiplication");
    };
    assert_eq!(operation.op, BinaryOp::MatMul);
}

#[test]
fn parses_trait_scoped_behavior_pairs() {
    let source = concat!(
        "trait Metric:\n",
        "    @metric\n",
        "    with(context):\n",
        "        context.start()\n",
        "    without(context):\n",
        "        context.finish()\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Trait(metric) = &module.items[0] else {
        panic!("expected Metric trait");
    };
    assert_eq!(metric.scoped_behaviors.len(), 2);
    assert_eq!(
        metric.scoped_behaviors[0].phase,
        severian_ast::TraitScopedBehaviorPhase::With
    );
    assert_eq!(
        metric.scoped_behaviors[1].phase,
        severian_ast::TraitScopedBehaviorPhase::Without
    );
    assert_eq!(metric.scoped_behaviors[0].params[0].name.name, "context");
}

#[test]
fn parses_required_and_defaulted_trait_registry_properties() {
    let source = concat!(
        "trait File:\n",
        "    property file_type: FileType\n",
        "    property extensions: set[string] = {\".txt\"}\n",
        "    def read(path: string) -> string\n",
        "\n",
        "class TextFile: File:\n",
        "    file_type = FileType.TXT\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Trait(file) = &module.items[0] else {
        panic!("expected File trait");
    };
    assert_eq!(file.properties.len(), 2);
    assert_eq!(file.properties[0].name.name, "file_type");
    assert!(file.properties[0].default.is_none());
    assert_eq!(file.properties[1].name.name, "extensions");
    assert!(file.properties[1].default.is_some());
    let Item::Class(text_file) = &module.items[1] else {
        panic!("expected TextFile class");
    };
    assert_eq!(text_file.traits.len(), 1);
}

#[test]
fn parses_bitwise_precedence_below_comparisons_and_arithmetic() {
    let source =
        "def combine(a: int, b: int, c: int) -> bool:\n    return a + 1 & b ^ c | a == b\n";
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::Return(return_) = &function.body.statements[0] else {
        panic!("expected return");
    };
    let Some(Expr::Binary(equal)) = &return_.value else {
        panic!("expected equality at the root");
    };
    assert_eq!(equal.op, severian_ast::BinaryOp::Equal);
    let Expr::Binary(bit_or) = equal.left.as_ref() else {
        panic!("expected bitwise-or below equality");
    };
    assert_eq!(bit_or.op, severian_ast::BinaryOp::BitOr);
}

#[test]
fn parses_multiple_generic_class_arguments_as_a_type_tuple() {
    let source = concat!(
        "class Pair[Left, Right]:\n",
        "    left: Left\n",
        "    right: Right\n",
        "\n",
        "def pair() -> Pair[int, string]:\n",
        "    return Pair[int, string](1, \"one\")\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[1] else {
        panic!("expected function");
    };
    let Stmt::Return(return_) = &function.body.statements[0] else {
        panic!("expected return");
    };
    let Some(Expr::Call(call)) = &return_.value else {
        panic!("expected constructor call");
    };
    let Expr::Index(index) = call.callee.as_ref() else {
        panic!("expected generic class application");
    };
    let Expr::Tuple(arguments) = index.index.as_ref() else {
        panic!("expected multiple type arguments");
    };
    assert_eq!(arguments.elements.len(), 2);
}

#[test]
fn parses_cross_field_class_invariants() {
    let source = concat!(
        "class Range:\n",
        "    low: int with { low >= 0 }\n",
        "    high: int = 10 with { high > low, high < 100 }\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Class(class) = &module.items[0] else {
        panic!("expected class");
    };
    assert_eq!(class.fields[0].constraints.len(), 1);
    assert_eq!(class.fields[1].constraints.len(), 2);
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
fn keeps_result_capture_distinct_from_assignment() {
    let source = concat!(
        "def main():\n",
        "    outcome ?= read(\"settings.toml\")\n",
        "    value = read(\"settings.toml\")\n",
        "    changing := read(\"settings.toml\")\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    assert!(matches!(function.body.statements[0], Stmt::TryBind(_)));
    assert!(matches!(function.body.statements[1], Stmt::Let(_)));
    assert!(matches!(function.body.statements[2], Stmt::Let(_)));
}

#[test]
fn parses_an_explicit_extern_abi_declaration() {
    let source = concat!(
        "unsafe:\n",
        "    extern(\"__sev_file_read\") def fileRead(\n",
        "        path: string,\n",
        "    ) -> Result[string, IOError]\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected extern function declaration");
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
fn extern_abi_parameters_must_remain_explicitly_typed() {
    let source = "unsafe:\n    extern(\"host_value\") def hostValue(value) -> int\n";
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "extern ABI parameters require explicit types"
    );
}

#[test]
fn rejects_an_extern_abi_declaration_without_unsafe() {
    let error = parse(&lex("extern(\"host_call\") def hostCall()\n").unwrap()).unwrap_err();

    assert_eq!(
        error.message,
        "extern declarations cross the host ABI and require an `unsafe:` block"
    );
}

#[test]
fn rejects_an_inline_unsafe_extern_declaration() {
    let error = parse(&lex("unsafe extern(\"host_call\") def hostCall()\n").unwrap()).unwrap_err();

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

#[test]
fn parses_switch_alternatives_and_python_match_cases() {
    let source = concat!(
        "def classify(extension: string) -> int:\n",
        "    match extension:\n",
        "        case \".yaml\" | \".yml\":\n",
        "            return 1\n",
        "        case _:\n",
        "            return 0\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected function");
    };
    let Stmt::Switch(statement) = &function.body.statements[0] else {
        panic!("expected match to lower to a switch statement");
    };
    assert_eq!(statement.arms.len(), 3);
}

#[test]
fn keeps_match_available_as_a_function_name() {
    let source = concat!(
        "def match(pattern: string, text: string) -> bool:\n",
        "    return pattern == text\n",
        "\n",
        "def main():\n",
        "    match(\"same\", \"same\")\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[1] else {
        panic!("expected main function");
    };
    assert!(matches!(
        function.body.statements[0],
        Stmt::Expr(Expr::Call(_))
    ));
}

#[test]
fn parses_function_declarations_inside_function_bodies() {
    let source = concat!(
        "def outer(value: int) -> int:\n",
        "    def inner(offset: int) -> int:\n",
        "        return value + offset\n",
        "    return inner(2)\n",
    );
    let module = parse(&lex(source).unwrap()).unwrap();
    let Item::Function(function) = &module.items[0] else {
        panic!("expected outer function");
    };
    assert!(matches!(function.body.statements[0], Stmt::Function(_)));
}
