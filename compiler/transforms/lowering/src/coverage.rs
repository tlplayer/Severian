use crate::location::MlirLocation;
use severian_coverage::{
    source_map::{stable_region_id, CoverageRegionKind},
    CoverageRegion, CoverageRegionId, CoverageSourceMap, SourcePosition,
    SourceSpan as CoverageSpan,
};
use severian_source::{SourceMap, SourceSpan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageGranularity {
    Function,
    Statement,
    Branch,
    Full,
}

#[derive(Debug, Clone)]
pub struct CoverageLoweringOptions {
    pub enabled: bool,
    pub granularity: CoverageGranularity,
}

impl Default for CoverageLoweringOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            granularity: CoverageGranularity::Full,
        }
    }
}

pub struct CoverageLowering<'a> {
    sources: &'a SourceMap,
    map: CoverageSourceMap,
}

impl<'a> CoverageLowering<'a> {
    pub fn new(sources: &'a SourceMap) -> Self {
        Self {
            sources,
            map: CoverageSourceMap::default(),
        }
    }

    pub fn record(
        &mut self,
        function: &str,
        span: SourceSpan,
        kind: CoverageRegionKind,
    ) -> Option<(CoverageRegionId, MlirLocation)> {
        let file = self.sources.file(span.file)?;
        let start = file.location(span.bytes.start)?;
        let end = file.location(span.bytes.end)?;

        let coverage_span = CoverageSpan {
            file: file.path().to_owned(),
            start: SourcePosition {
                line: start.line,
                column: start.column,
                byte: start.byte,
            },
            end: SourcePosition {
                line: end.line,
                column: end.column,
                byte: end.byte,
            },
        };

        let id = stable_region_id(function, &coverage_span, kind);
        self.map.insert(CoverageRegion {
            id,
            function: function.to_owned(),
            span: coverage_span,
            kind,
        });

        Some((id, MlirLocation::from_source(self.sources, span)))
    }

    pub fn finish(self) -> CoverageSourceMap {
        self.map
    }
}

pub fn required_llvm_runtime_symbols() -> &'static [&'static str] {
    &["__llvm_profile_runtime", "__llvm_profile_register_function"]
}
