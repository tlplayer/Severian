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
}
