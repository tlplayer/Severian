#![forbid(unsafe_code)]

mod statement;
pub use statement::{parse, parse_with_max_errors};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_lexer::scan;
    use severian_source::SourceFile;

    #[test]
    fn parses_two_bindings() {
        let source = SourceFile::virtual_source("test.sev", "b = 2, a = 1 + b");
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(module.items.len(), 2);
        assert!(module
            .items
            .iter()
            .all(|item| matches!(item, severian_ast::Item::Binding(_))));
    }

    #[test]
    fn multiline_delimiters_ignore_layout_tokens_and_allow_trailing_commas() {
        let source = SourceFile::virtual_source(
            "multiline.sev",
            "def matrix(\n    rows: list[int],\n) -> list[int]:\n    return build(\n        [1, 2,\n         3, 4,\n        ],\n    )\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        assert_eq!(function.parameters.len(), 1);
        assert!(function.body.is_some());
    }

    #[test]
    fn compound_assignment_is_an_explicit_binding_update() {
        let source = SourceFile::virtual_source("update.sev", "value := 1\nvalue += 2\n");
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Binding(declaration) = &module.items[0] else {
            panic!("expected mutable declaration")
        };
        let severian_ast::Item::Binding(update) = &module.items[1] else {
            panic!("expected binding update")
        };
        assert!(declaration.mutable);
        assert!(update.update);
        assert!(matches!(
            update.value.kind,
            severian_ast::ExpressionKind::Binary {
                operator: severian_ast::BinaryOperator::Add,
                ..
            }
        ));
    }

    #[test]
    fn question_equal_preserves_the_result_union() {
        let source = SourceFile::virtual_source("result.sev", "result ?= read()\n");
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Binding(binding) = &module.items[0] else {
            panic!("expected an error-preserving binding")
        };
        assert!(binding.preserve_error);
        assert!(!binding.update);
    }

    #[test]
    fn try_catch_preserves_the_error_binding() {
        let source = SourceFile::virtual_source(
            "catch.sev",
            "def main():\n    try:\n        fail()\n    catch error:\n        print(error.message)\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected a function")
        };
        let body = function.body.as_ref().unwrap();
        assert!(matches!(
            &body[0],
            severian_ast::Statement::Try {
                catch_binding,
                catch_annotation: None,
                ..
            } if catch_binding == "error"
        ));
    }

    #[test]
    fn invalid_expression_reports_a_label_note_and_help() {
        let source = SourceFile::virtual_source("invalid.sev", "value = .\n");
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.code, "E000111");
        assert_eq!(error.labels.len(), 1);
        assert!(!error.notes.is_empty());
        assert!(error.help.as_deref().unwrap().contains("`0.5`"));
    }

    #[test]
    fn prefix_typed_constants_use_the_same_binding_ast() {
        let source = SourceFile::virtual_source(
            "constants.sev",
            "int MAX_RETRIES = 3\nfloat PI = 3.1415926\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let module = parse(&tokens).unwrap();
        let severian_ast::Item::Binding(first) = &module.items[0] else {
            panic!("expected a binding")
        };
        assert_eq!(first.name, "MAX_RETRIES");
        assert!(!first.mutable);
        assert_eq!(
            first.annotation.as_ref().unwrap().simple_name(),
            Some("int")
        );
        let severian_ast::Item::Binding(second) = &module.items[1] else {
            panic!("expected a binding")
        };
        assert_eq!(second.name, "PI");
        assert_eq!(
            second.annotation.as_ref().unwrap().simple_name(),
            Some("float")
        );
    }

    #[test]
    fn semicolon_separates_without_starting_a_physical_line() {
        let source = SourceFile::virtual_source("test.sev", "x = 1; y = 2");
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(module.items.len(), 2);
    }

    #[test]
    fn formatted_block_strings_desugar_to_string_concatenation() {
        let source = SourceFile::virtual_source(
            "formatted.sev",
            r#"body = "content"
output = f"""module {{
{body}}}
"""
"#,
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Binding(output) = &module.items[1] else {
            panic!("expected output binding")
        };
        let severian_ast::ExpressionKind::Binary { operator, .. } = &output.value.kind else {
            panic!("formatted string was not desugared")
        };
        assert_eq!(*operator, severian_ast::BinaryOperator::Add);
    }

    #[test]
    fn formatted_block_strings_reject_unescaped_closing_braces() {
        let source = SourceFile::virtual_source("formatted.sev", "value = f\"\"\"bad }\"\"\"\n");
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.code, "E000113");
    }

    #[test]
    fn power_is_right_associative() {
        let source = SourceFile::virtual_source("test.sev", "x = 2 ** 3 ** 2");
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Binding(binding) = &module.items[0] else {
            unreachable!()
        };
        let severian_ast::ExpressionKind::Binary { right, .. } = &binding.value.kind else {
            unreachable!()
        };
        assert!(matches!(
            right.kind,
            severian_ast::ExpressionKind::Binary {
                operator: severian_ast::BinaryOperator::Power,
                ..
            }
        ));
    }

    #[test]
    fn parses_task_ownership_locking_and_await() {
        let source = SourceFile::virtual_source(
            "tasks.sev",
            "def work(value: int) -> int:\n    return value\n\ndef main():\n    task = async work(1) with self and lock\n    result = await task\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(main) = &module.items[1] else {
            panic!("expected main")
        };
        let body = main.body.as_ref().unwrap();
        let severian_ast::Statement::Binding(task) = &body[0] else {
            panic!("expected task binding")
        };
        assert!(matches!(
            task.value.kind,
            severian_ast::ExpressionKind::Async {
                owner: severian_ast::TaskOwner::SelfScope,
                locked: true,
                ..
            }
        ));
        let severian_ast::Statement::Binding(result) = &body[1] else {
            panic!("expected result binding")
        };
        assert!(matches!(
            result.value.kind,
            severian_ast::ExpressionKind::Await { .. }
        ));
    }

    #[test]
    fn parses_entry_and_deferred_function_contracts() {
        let source = SourceFile::virtual_source(
            "contracts.sev",
            "def bounded(value: int) -> int with\n{\n    value >= 0,\n    defer value <= 10 -> Error(\"too large\"),\n}:\n    return value\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        assert_eq!(function.contracts.len(), 2);
        assert!(!function.contracts[0].deferred);
        assert!(function.contracts[1].deferred);
        assert!(function.contracts[1].failure.is_some());
    }

    #[test]
    fn parses_hook_signatures_and_structured_phases_separately_from_contracts() {
        let source = SourceFile::virtual_source(
            "hooks.sev",
            "trait Monitor:\n    @monitor_error\n    def monitor_error(context: string) -> None with context\nclass Metric: Monitor\n    def monitor_error(context: string) -> None with context:\n        with context:\n            print(\"start\")\n        without context:\n            print(\"finish\")\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Trait(declaration) = &module.items[0] else {
            panic!("expected trait")
        };
        let required = &declaration.methods[0];
        assert!(required.contracts.is_empty());
        let hook = required.hook.as_ref().expect("expected hook signature");
        assert_eq!(hook.context, "context");
        assert!(hook.with_phase.is_empty());
        assert!(hook.without_phase.is_empty());

        let severian_ast::Item::Class(declaration) = &module.items[1] else {
            panic!("expected class")
        };
        let implementation = &declaration.methods[0];
        assert!(implementation.contracts.is_empty());
        let hook = implementation
            .hook
            .as_ref()
            .expect("expected hook implementation");
        assert_eq!(hook.with_phase.len(), 1);
        assert_eq!(hook.without_phase.len(), 1);
        assert_eq!(implementation.body.as_deref(), Some([].as_slice()));
    }

    #[test]
    fn hook_phases_must_use_the_declared_context() {
        let source = SourceFile::virtual_source(
            "invalid-hook.sev",
            "def trace(context: string) with context:\n    with other:\n        print(\"start\")\n",
        );
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.code, "E000112");
        assert!(error.message.contains("expected `context`"));
    }

    #[test]
    fn parses_primitive_trait_contract() {
        let source = SourceFile::virtual_source(
            "i32.sev",
            "trait i32: Primitive + Integer[i32]:\n    property category: string = \"integer\"\n    operator +(right: i32) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let declaration = module
            .items
            .iter()
            .find_map(|item| match item {
                severian_ast::Item::Trait(declaration) => Some(declaration),
                _ => None,
            })
            .unwrap();
        assert_eq!(declaration.name, "i32");
        assert_eq!(declaration.bases.len(), 2);
        assert_eq!(
            declaration.operators[0].operator,
            severian_ast::OperatorSyntax::Plus
        );
    }

    #[test]
    fn parses_canonical_trait_methods_and_composed_traits() {
        let source = SourceFile::virtual_source(
            "drawable.sev",
            "trait Named:\n    def name() -> string\n\ntrait Drawable:\n    Named\n    def draw()\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Trait(drawable) = &module.items[1] else {
            panic!("expected Drawable trait")
        };
        assert_eq!(drawable.bases[0].simple_name(), Some("Named"));
        assert_eq!(drawable.methods[0].name, "draw");
        assert!(drawable.methods[0].body.is_none());
    }

    #[test]
    fn rejects_trait_method_shorthand() {
        let source = SourceFile::virtual_source("invalid.sev", "trait Drawable:\n    draw()\n");
        assert!(parse(&scan(&source).unwrap()).is_err());
    }

    #[test]
    fn parses_class_fields_constructors_methods_and_traits() {
        let source = SourceFile::virtual_source(
            "point.sev",
            "class Point[T]: Drawable[T] + Copy\n    x: T\n    y: T = 0\n\n    def Point(x: T, y: T):\n        pass\n\n    def draw():\n        pass\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Class(point) = &module.items[0] else {
            panic!("expected Point class")
        };
        assert_eq!(point.type_parameters, ["T"]);
        assert_eq!(point.traits.len(), 2);
        assert_eq!(point.fields.len(), 2);
        assert_eq!(point.constructors.len(), 1);
        assert_eq!(point.methods.len(), 1);
    }

    #[test]
    fn parses_generic_class_construction_and_field_update_methods() {
        let source = SourceFile::virtual_source(
            "box.sev",
            "class Box[T]:\n    value: T\n    def addition(addition: T):\n        value += addition\ndef main():\n    boxed := Box[int](10)\n    boxed.addition(20)\n    print(boxed.value)\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(main) = &module.items[1] else {
            panic!("expected main")
        };
        let severian_ast::Statement::Binding(binding) = &main.body.as_ref().unwrap()[0] else {
            panic!("expected constructed binding")
        };
        let severian_ast::ExpressionKind::Call { callee, .. } = &binding.value.kind else {
            panic!("expected constructor call")
        };
        assert!(matches!(
            callee.kind,
            severian_ast::ExpressionKind::TypeApplication { .. }
        ));
    }

    #[test]
    fn trait_free_class_keeps_the_terminal_colon() {
        let source = SourceFile::virtual_source("point.sev", "class Point:\n    x: int\n");
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert!(matches!(module.items[0], severian_ast::Item::Class(_)));
    }

    #[test]
    fn implemented_trait_header_rejects_a_terminal_colon() {
        let source =
            SourceFile::virtual_source("invalid.sev", "class Point: Drawable:\n    x: int\n");
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert!(error.message.contains("do not take a trailing `:`"));
    }

    #[test]
    fn one_recursive_type_parser_serves_every_annotation_position() {
        let source = SourceFile::virtual_source(
            "types.sev",
            "trait Example: Base[Tensor[f32]]:\n    property value: list[Tensor[f16]]\n    operator +(right: F[int, string]) -> int | None\nx: list[Tensor[f16]] = 1\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert!(matches!(module.items[0], severian_ast::Item::Trait(_)));
        assert!(matches!(module.items[1], severian_ast::Item::Binding(_)));
        let severian_ast::Item::Trait(declaration) = &module.items[0] else {
            unreachable!()
        };
        let severian_ast::Item::Binding(binding) = &module.items[1] else {
            unreachable!()
        };
        let base = &declaration.bases[0];
        let property = &declaration.properties[0].annotation;
        let parameter = &declaration.operators[0].parameters[0].annotation;
        let result = &declaration.operators[0].result;
        let binding = binding.annotation.as_ref().unwrap();
        assert!(matches!(
            base.kind,
            severian_ast::TypeAnnotationKind::Named { .. }
        ));
        assert!(matches!(
            property.kind,
            severian_ast::TypeAnnotationKind::Named { .. }
        ));
        assert!(matches!(
            parameter.kind,
            severian_ast::TypeAnnotationKind::Named { .. }
        ));
        assert!(matches!(
            result.kind,
            severian_ast::TypeAnnotationKind::Union(_)
        ));
        assert_eq!(binding.named_parts().unwrap().0, "list");
        assert_eq!(property.named_parts().unwrap().0, "list");
        assert_eq!(binding.named_parts().unwrap().1.len(), 1);
        assert_eq!(property.named_parts().unwrap().1.len(), 1);
    }

    #[test]
    fn generic_bounds_and_with_constraints_share_one_ast() {
        let source = SourceFile::virtual_source(
            "constraints.sev",
            "def process[T: Ordered, N: usize](value: T) -> T with {\n    T: Clone,\n    N > 0,\n}:\n    return value\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        assert_eq!(function.type_parameters, ["T", "N"]);
        assert_eq!(function.constraints.len(), 4);
        assert!(matches!(
            function.constraints[0],
            severian_ast::GenericConstraint::Parameter { .. }
        ));
        assert!(matches!(
            function.constraints[3],
            severian_ast::GenericConstraint::Predicate(_)
        ));
    }

    #[test]
    fn parses_intersection_bounds_and_map_loop_bindings() {
        let source = SourceFile::virtual_source(
            "map-generic.sev",
            "def collect[K: Hash + Equal, V: Number](values: map[K, V]):\n    for key, value in values:\n        pass\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        assert_eq!(function.constraints.len(), 3);
        let body = function.body.as_ref().unwrap();
        let severian_ast::Statement::For {
            binding,
            second_binding,
            ..
        } = &body[0]
        else {
            panic!("expected for statement")
        };
        assert_eq!(binding, "key");
        assert_eq!(second_binding.as_deref(), Some("value"));
    }

    #[test]
    fn parses_decorated_boundary_declarations_as_normal_declarations() {
        let source = SourceFile::virtual_source(
            "ffi.sev",
            "@c(symbol = \"strlen\")\ndef length(value: borrowed[string]) -> usize\n@rust\ntype RustBuffer[T]\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            unreachable!()
        };
        assert_eq!(function.decorators[0].name, "c");
        assert_eq!(
            function.parameters[0].annotation.named_parts().unwrap().0,
            "borrowed"
        );
        assert!(matches!(module.items[1], severian_ast::Item::Type(_)));
    }

    #[test]
    fn language_attribute_replaces_the_extern_keyword() {
        let source =
            SourceFile::virtual_source("ffi.sev", "@c\nextern def legacy(value: i32) -> i32\n");
        assert!(parse(&scan(&source).unwrap()).is_err());
    }

    #[test]
    fn parses_ordered_global_calls_and_an_optional_main_body() {
        let source = SourceFile::virtual_source(
            "entry.sev",
            "print(\"global\")\nseed := 7\ndef main():\n    print(seed)\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert!(matches!(module.items[0], severian_ast::Item::Expression(_)));
        assert!(matches!(module.items[1], severian_ast::Item::Binding(_)));
        let severian_ast::Item::Function(main) = &module.items[2] else {
            unreachable!()
        };
        assert_eq!(main.name, "main");
        assert_eq!(main.body.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn parses_named_ordinary_and_composed_test_declarations() {
        let source = SourceFile::virtual_source(
            "tests.sev",
            "test:\n    assert(true)\n\ntest with property and chaos \"generated\":\n    assert(true)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let module = parse(&tokens).unwrap();
        let tests = module
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Test(test) => Some(test),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tests.len(), 2);
        assert_eq!(tests[0].name, None);
        assert_eq!(tests[1].name.as_deref(), Some("generated"));
        assert_eq!(tests[1].modes, ["property", "chaos"]);
    }

    #[test]
    fn match_cases_use_bindings_and_types_instead_of_magic_names() {
        let source = SourceFile::virtual_source(
            "match.sev",
            "def handle(result: int) -> int:\n    match result:\n        case error: int:\n            return error\n        case int failure:\n            return failure\n        case _:\n            return 0\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let module = parse(&tokens).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        let severian_ast::Statement::Match { cases, .. } = &function.body.as_ref().unwrap()[0]
        else {
            panic!("expected match")
        };
        assert_eq!(cases[0].binding.as_deref(), Some("error"));
        assert_eq!(
            cases[0].annotation.as_ref().unwrap().simple_name(),
            Some("int")
        );
        assert_eq!(cases[1].binding.as_deref(), Some("failure"));
        assert_eq!(
            cases[1].annotation.as_ref().unwrap().simple_name(),
            Some("int")
        );
        assert_eq!(cases[2].binding, None);
        assert_eq!(cases[2].annotation, None);
    }

    #[test]
    fn rejects_function_control_flow_at_global_scope() {
        let source = SourceFile::virtual_source("invalid.sev", "return 1\n");
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.code, "E000121");
    }

    #[test]
    fn parses_fallback_throw_and_throws_expectation_forms() {
        let source = SourceFile::virtual_source(
            "optional.sev",
            "test:\n    value = maybe() else \"fallback\"\n    throws(maybe() else throw Error(\"missing\"))\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Test(test) = &module.items[0] else {
            panic!("expected test")
        };
        let severian_ast::Statement::Binding(binding) = &test.body[0] else {
            panic!("expected fallback binding")
        };
        assert!(matches!(
            binding.value.kind,
            severian_ast::ExpressionKind::Fallback { .. }
        ));
        let severian_ast::Statement::Expression(expectation) = &test.body[1] else {
            panic!("expected throws expression")
        };
        assert!(matches!(
            expectation.kind,
            severian_ast::ExpressionKind::Call { .. }
        ));
    }

    #[test]
    fn reports_multiple_syntax_errors_from_one_source() {
        let source = SourceFile::virtual_source(
            "invalid.sev",
            "def main():\n    first = )\n    second = )\n",
        );
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.additional.len(), 1, "{error:#?}");
        let rendered = error.with_source(source).to_string();
        assert!(rendered.contains("invalid.sev:2"), "{rendered}");
        assert!(rendered.contains("invalid.sev:3"), "{rendered}");
    }

    #[test]
    fn compound_field_assignment_preserves_the_field_read_and_update() {
        let source = SourceFile::virtual_source(
            "field-update.sev",
            "def update(counter: Counter):\n    counter.value += 1\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Function(function) = &module.items[0] else {
            panic!("expected function")
        };
        let severian_ast::Statement::FieldAssignment { field, value, .. } =
            &function.body.as_ref().unwrap()[0]
        else {
            panic!("expected field assignment")
        };
        assert_eq!(field, "value");
        assert!(matches!(
            value.kind,
            severian_ast::ExpressionKind::Binary {
                operator: severian_ast::BinaryOperator::Add,
                ..
            }
        ));
    }

    #[test]
    fn parses_field_constraints_maps_chained_comparisons_and_typed_throws() {
        let source = SourceFile::virtual_source(
            "dynamic-fields.sev",
            "class Point:\n    x: int {0 <= x <= 100}\n    y: int\n\ntest:\n    point := Point(1, 2)\n    point.set({\n        \"x\": 20,\n        \"y\": 30,\n    })\n    throws(point.get(\"z\") -> FieldError)\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Class(class) = &module.items[0] else {
            panic!("expected class")
        };
        let constraint = &class.fields[0].constraints[0].condition;
        assert!(matches!(
            constraint.kind,
            severian_ast::ExpressionKind::Binary {
                operator: severian_ast::BinaryOperator::And,
                ..
            }
        ));
        let severian_ast::Item::Test(test) = &module.items[1] else {
            panic!("expected test")
        };
        let severian_ast::Statement::Expression(set) = &test.body[1] else {
            panic!("expected set call")
        };
        let severian_ast::ExpressionKind::Call { arguments, .. } = &set.kind else {
            panic!("expected call")
        };
        assert!(matches!(arguments[0].value.kind, severian_ast::ExpressionKind::Map(_)));
        let severian_ast::Statement::Expression(expectation) = &test.body[2] else {
            panic!("expected throws call")
        };
        let severian_ast::ExpressionKind::Call { arguments, .. } = &expectation.kind else {
            panic!("expected call")
        };
        assert!(matches!(
            arguments[0].expected_error.as_ref().unwrap().kind,
            severian_ast::ExpressionKind::Name(ref name) if name == "FieldError"
        ));
    }

    #[test]
    fn parses_ordered_property_failures_builder_continuations_and_mock_cases() {
        let source = SourceFile::virtual_source(
            "builders.sev",
            "class User:\n    age: int {\n        age < 0\n            -> Error(\"negative\"),\n        age > 130\n            -> invalid_age(age),\n    }\n\ntest:\n    user := User()\n        .set(age, 36)\n    mock(\n        foo(0) -> 10,\n        foo(1) -> 20,\n        else throw Error(\"unexpected\")\n    )\n    throws(foo(2) -> Error(\"unexpected\"))\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Class(class) = &module.items[0] else {
            panic!("expected class")
        };
        assert_eq!(class.fields[0].constraints.len(), 2);
        assert!(class.fields[0]
            .constraints
            .iter()
            .all(|constraint| constraint.failure.is_some()));
        let severian_ast::Item::Test(test) = &module.items[1] else {
            panic!("expected test")
        };
        let severian_ast::Statement::Binding(builder) = &test.body[0] else {
            panic!("expected builder binding")
        };
        assert!(matches!(builder.value.kind, severian_ast::ExpressionKind::Call { .. }));
        let severian_ast::Statement::Expression(mock) = &test.body[1] else {
            panic!("expected mock")
        };
        assert!(matches!(
            mock.kind,
            severian_ast::ExpressionKind::Mock { ref cases, .. } if cases.len() == 2
        ));
    }

    #[test]
    fn parses_trait_owned_pipe_operators_and_compiler_case_functions() {
        let source = SourceFile::virtual_source(
            "operators.sev",
            "trait StringOperator:\n    @strings\n\n    operator |(left: string, right: string) -> string\nclass Strings:\n    trait StringOperator\n    operator |(left: string, right: string) -> string:\n        return left + right\n@strings(|)\ndef combine(left: string, right: string) -> string:\n    return left | right\ntest with compiler:\n    reject:\n        @strings(|)\n        @other(|)\n        def ambiguous(left: string, right: string) -> string:\n            return left | right\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Trait(trait_declaration) = &module.items[0] else {
            panic!("expected trait")
        };
        assert!(trait_declaration.methods.is_empty());
        assert_eq!(
            trait_declaration.operators[0].operator,
            severian_ast::OperatorSyntax::Pipe
        );
        assert_eq!(trait_declaration.namespaces[0].name, "strings");
        assert!(trait_declaration.operators[0].decorators.is_empty());
        let severian_ast::Item::Class(class) = &module.items[1] else {
            panic!("expected class")
        };
        assert_eq!(class.traits[0].simple_name(), Some("StringOperator"));
        assert!(class.methods.is_empty());
        assert_eq!(class.operators[0].operator, severian_ast::OperatorSyntax::Pipe);
        assert_eq!(class.operators[0].body.len(), 1);
        let severian_ast::Item::Function(function) = &module.items[2] else {
            panic!("expected function")
        };
        assert!(matches!(
            function.body.as_ref().unwrap()[0],
            severian_ast::Statement::Return {
                value: Some(severian_ast::Expression {
                    kind: severian_ast::ExpressionKind::Binary {
                        operator: severian_ast::BinaryOperator::Pipe,
                        ..
                    },
                    ..
                }),
                ..
            }
        ));
        let severian_ast::Item::Test(test) = &module.items[3] else {
            panic!("expected test")
        };
        assert_eq!(test.compiler_cases[0].items.len(), 1);
    }

    #[test]
    fn separates_trait_namespaces_from_member_decorators() {
        let source = SourceFile::virtual_source(
            "hooks.sev",
            "trait Monitor:\n    @monitor\n\n    @monitor_error\n    def monitor_error(context: HookContext) -> None with context\n\n    @monitor_time\n    def monitor_time(context: HookContext) -> None with context\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        let severian_ast::Item::Trait(declaration) = &module.items[0] else {
            panic!("expected trait")
        };
        assert_eq!(declaration.namespaces[0].name, "monitor");
        assert_eq!(declaration.methods[0].decorators[0].name, "monitor_error");
        assert_eq!(declaration.methods[1].decorators[0].name, "monitor_time");
    }

    #[test]
    fn symbolic_operators_are_not_function_names() {
        let source = SourceFile::virtual_source(
            "invalid-operator.sev",
            "trait Invalid:\n    def |(left: string, right: string) -> string\n",
        );
        let error = parse(&scan(&source).unwrap()).unwrap_err();
        assert_eq!(error.code, "E000110");
        assert!(error.message.contains("function name"));
    }
}
