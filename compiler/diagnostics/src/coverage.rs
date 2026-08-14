use crate::{Diagnostic, DiagnosticBag};

#[derive(Debug, Clone, Copy)]
pub struct CoverageThresholds {
    pub lines: Option<f64>,
    pub regions: Option<f64>,
    pub branches: Option<f64>,
    pub functions: Option<f64>,
}

impl Default for CoverageThresholds {
    fn default() -> Self {
        Self {
            lines: None,
            regions: None,
            branches: None,
            functions: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoveragePercentages {
    pub lines: f64,
    pub regions: f64,
    pub branches: f64,
    pub functions: f64,
}

pub fn check_thresholds(
    actual: CoveragePercentages,
    required: CoverageThresholds,
) -> DiagnosticBag {
    let mut bag = DiagnosticBag::default();

    check(&mut bag, "lines", actual.lines, required.lines);
    check(&mut bag, "regions", actual.regions, required.regions);
    check(&mut bag, "branches", actual.branches, required.branches);
    check(&mut bag, "functions", actual.functions, required.functions);

    bag
}

fn check(bag: &mut DiagnosticBag, category: &str, actual: f64, required: Option<f64>) {
    let Some(required) = required else {
        return;
    };

    if actual + f64::EPSILON < required {
        bag.push(
            Diagnostic::error(
                "coverage::threshold",
                format!(
                    "{category} coverage is {actual:.2}% but the required threshold is {required:.2}%"
                ),
            )
            .with_help("add tests or lower the configured threshold intentionally"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_every_coverage_metric_at_the_exact_boundary() {
        let required = CoverageThresholds {
            lines: Some(99.0),
            regions: Some(99.0),
            branches: Some(99.0),
            functions: Some(99.0),
        };
        assert!(!check_thresholds(
            CoveragePercentages {
                lines: 99.0,
                regions: 99.0,
                branches: 99.0,
                functions: 99.0,
            },
            required,
        )
        .has_errors());

        let diagnostics = check_thresholds(
            CoveragePercentages {
                lines: 98.999,
                regions: 98.999,
                branches: 98.999,
                functions: 98.999,
            },
            required,
        );
        assert_eq!(diagnostics.error_count(), 4);
        assert!(diagnostics.has_errors());
    }
}
