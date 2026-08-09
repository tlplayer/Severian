use severian_ast::Span;
use severian_hir::source::{HirSourceKey, HirSourceMap};
use severian_source::{FileId, SourceMap, SourceMapError, SourceSpan};

pub struct SemanticSourceRecorder<'a> {
    source_map: &'a SourceMap,
    file: FileId,
    hir: HirSourceMap,
}

impl<'a> SemanticSourceRecorder<'a> {
    pub fn new(source_map: &'a SourceMap, file: FileId) -> Self {
        Self { source_map, file, hir: HirSourceMap::new() }
    }

    pub fn finish(self) -> HirSourceMap { self.hir }

    pub fn record(
        &mut self,
        key: HirSourceKey,
        span: Span,
    ) -> Result<SourceSpan, SourceMapError> {
        let span = self.source_map.from_ast_span(self.file, span)?;
        self.hir.insert(key, span);
        Ok(span)
    }

    pub fn record_function(
        &mut self,
        function: &str,
        span: Span,
    ) -> Result<SourceSpan, SourceMapError> {
        self.record(HirSourceKey::Function(function.to_owned()), span)
    }

    pub fn record_parameter(
        &mut self,
        function: &str,
        parameter: &str,
        span: Span,
    ) -> Result<SourceSpan, SourceMapError> {
        self.record(
            HirSourceKey::Parameter {
                function: function.to_owned(),
                parameter: parameter.to_owned(),
            },
            span,
        )
    }

    pub fn record_instruction(
        &mut self,
        function: &str,
        path: &[usize],
        span: Span,
    ) -> Result<SourceSpan, SourceMapError> {
        self.record(
            HirSourceKey::Instruction {
                function: function.to_owned(),
                path: path.to_vec(),
            },
            span,
        )
    }

    pub fn record_expression(
        &mut self,
        function: &str,
        instruction_path: &[usize],
        expression_path: &[usize],
        span: Span,
    ) -> Result<SourceSpan, SourceMapError> {
        self.record(
            HirSourceKey::Expression {
                function: function.to_owned(),
                instruction_path: instruction_path.to_vec(),
                expression_path: expression_path.to_vec(),
            },
            span,
        )
    }
}
