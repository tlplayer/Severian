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
        assert_eq!(module.bindings.len(), 2);
    }

    #[test]
    fn parses_primitive_trait_contract() {
        let source = SourceFile::virtual_source(
            "i32.sev",
            "trait i32: Primitive + Integer[i32]:\n    property category: string = \"integer\"\n    operator +(right: i32) -> i32\n",
        );
        let module = parse(&scan(&source).unwrap()).unwrap();
        assert_eq!(module.traits[0].name, "i32");
        assert_eq!(module.traits[0].bases.len(), 2);
        assert_eq!(
            module.traits[0].operators[0].operator,
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
        let base = &module.traits[0].bases[0];
        let property = &module.traits[0].properties[0].annotation;
        let parameter = &module.traits[0].operators[0].parameters[0].annotation;
        let result = &module.traits[0].operators[0].result;
        let binding = module.bindings[0].annotation.as_ref().unwrap();
        assert!(matches!(base.kind, severian_ast::TypeAnnotationKind::Named { .. }));
        assert!(matches!(property.kind, severian_ast::TypeAnnotationKind::Named { .. }));
        assert!(matches!(parameter.kind, severian_ast::TypeAnnotationKind::Named { .. }));
        assert!(matches!(result.kind, severian_ast::TypeAnnotationKind::Union(_)));
        assert_eq!(binding.named_parts().unwrap().0, "list");
        assert_eq!(property.named_parts().unwrap().0, "list");
        assert_eq!(binding.named_parts().unwrap().1.len(), 1);
        assert_eq!(property.named_parts().unwrap().1.len(), 1);
    }
}
