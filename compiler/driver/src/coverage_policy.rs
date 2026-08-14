use severian_coverage::{CoverageReport, FileCoverageReport};
use severian_diagnostics::coverage::{check_thresholds, CoveragePercentages, CoverageThresholds};
use severian_diagnostics::{render, DiagnosticBag};
use severian_package::CoveragePolicy;

pub fn enforce(
    report: &CoverageReport,
    files: &[FileCoverageReport],
    policy: &CoveragePolicy,
) -> Result<(), String> {
    let thresholds = CoverageThresholds {
        lines: Some(policy.minimum),
        regions: policy.regions,
        branches: policy.branches,
        functions: policy.functions,
    };
    let aggregate_diagnostics = diagnostics(report, thresholds);
    let file_diagnostics = if policy.per_file {
        files
            .iter()
            .filter_map(|file| {
                let diagnostics = diagnostics(&file.report, thresholds);
                diagnostics
                    .has_errors()
                    .then_some((&file.file, diagnostics))
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    if !aggregate_diagnostics.has_errors() && file_diagnostics.is_empty() {
        return Ok(());
    }

    if aggregate_diagnostics.has_errors() {
        eprintln!("Aggregate coverage threshold failure:");
        render_diagnostics(&aggregate_diagnostics);
    }
    for (file, diagnostics) in &file_diagnostics {
        eprintln!("Per-file coverage threshold failure: {}", file.display());
        render_diagnostics(diagnostics);
    }
    Err(format!(
        "coverage thresholds were not met by the aggregate report or {} source file(s)",
        file_diagnostics.len()
    ))
}

fn diagnostics(report: &CoverageReport, thresholds: CoverageThresholds) -> DiagnosticBag {
    check_thresholds(
        CoveragePercentages {
            lines: report.lines.percent,
            regions: report.regions.percent,
            branches: report.branches.percent,
            functions: report.functions.percent,
        },
        thresholds,
    )
}

fn render_diagnostics(diagnostics: &DiagnosticBag) {
    eprintln!(
        "{}",
        render::render_bag(
            diagnostics,
            None,
            &render::RenderOptions {
                color: false,
                ..Default::default()
            },
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_coverage::CoverageMetric;
    use std::path::PathBuf;

    fn report(percentages: [f64; 4]) -> CoverageReport {
        let metric = |percent| CoverageMetric {
            count: 100,
            covered: percent as u64,
            percent,
        };
        CoverageReport {
            lines: metric(percentages[0]),
            regions: metric(percentages[1]),
            branches: metric(percentages[2]),
            functions: metric(percentages[3]),
            raw_json: None,
        }
    }

    #[test]
    fn per_file_policy_checks_lines_regions_branches_and_functions_independently() {
        let policy = CoveragePolicy {
            minimum: 99.0,
            changed_minimum: None,
            regions: Some(99.0),
            branches: Some(99.0),
            functions: Some(99.0),
            per_file: true,
        };
        let aggregate = report([100.0; 4]);

        for metric in 0..4 {
            let mut percentages = [100.0; 4];
            percentages[metric] = 98.0;
            let files = [FileCoverageReport {
                file: PathBuf::from("src/edge.sev"),
                report: report(percentages),
            }];
            assert!(enforce(&aggregate, &files, &policy).is_err());
        }

        let files = [FileCoverageReport {
            file: PathBuf::from("src/complete.sev"),
            report: report([99.0; 4]),
        }];
        assert!(enforce(&aggregate, &files, &policy).is_ok());
    }
}
