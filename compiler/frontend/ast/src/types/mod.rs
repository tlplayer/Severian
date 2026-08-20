use severian_source::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAnnotation {
    pub name: String,
    pub span: Span,
}
