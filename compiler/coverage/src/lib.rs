#![forbid(unsafe_code)]

pub mod instrument;
pub mod report;
pub mod source_map;

pub use instrument::{CoverageInstrumentationPlan, CoverageMode, CoverageProfileEnvironment};
pub use report::{
    export_report, language_report, merge_profiles, read_language_hits, render_files, CoverageMetric,
    CoverageReport, CoverageToolchain, FileCoverageReport,
};
pub use source_map::{
    CoverageRegion, CoverageRegionId, CoverageRegionKind, CoverageSourceMap, SourcePosition,
    SourceSpan,
};
