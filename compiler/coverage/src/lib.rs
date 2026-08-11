#![forbid(unsafe_code)]

pub mod instrument;
pub mod report;
pub mod source_map;

pub use instrument::{CoverageInstrumentationPlan, CoverageMode, CoverageProfileEnvironment};
pub use report::{
    export_report, merge_profiles, CoverageMetric, CoverageReport, CoverageToolchain,
};
pub use source_map::{
    CoverageRegion, CoverageRegionId, CoverageSourceMap, SourcePosition, SourceSpan,
};
