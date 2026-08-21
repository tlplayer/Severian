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
}
