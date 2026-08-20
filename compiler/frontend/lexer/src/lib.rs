#![forbid(unsafe_code)]

mod scanner;
mod token;

pub use scanner::scan;
pub use token::{Token, TokenKind};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    #[test]
    fn scans_integer_binding_and_addition() {
        let source = SourceFile::virtual_source("test.sev", "b = 2\na = 1 + b\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Plus));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Integer("2".into())));
    }

    #[test]
    fn scans_indented_trait_members_and_typed_operators() {
        let source = SourceFile::virtual_source(
            "primitive.sev",
            "trait i32: Primitive:\n    property bits: int = 32\n    operator +(right: i32) -> i32\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Indent));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Arrow));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Dedent));
    }
}
