#![forbid(unsafe_code)]

mod statement;
pub use statement::parse;

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
    fn semicolon_separates_without_starting_a_physical_line() {
        let source = SourceFile::virtual_source("test.sev", "x = 1; y = 2");
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(module.items.len(), 2);
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
}
