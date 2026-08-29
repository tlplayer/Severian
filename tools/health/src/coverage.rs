use crate::model::{Confidence, Evidence, Finding, Severity, SourceSpan};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_PERCENT: f64 = 95.0;

pub fn analyze(
    root: &Path,
    report: &Path,
    changed: Option<&BTreeMap<PathBuf, BTreeSet<usize>>>,
) -> Result<Vec<Finding>, String> {
    let source = fs::read_to_string(report).map_err(|error| {
        format!(
            "could not read coverage report {}: {error}",
            report.display()
        )
    })?;
    let document: Value = serde_json::from_str(&source)
        .map_err(|error| format!("invalid coverage JSON {}: {error}", report.display()))?;
    let mut findings = Vec::new();
    analyze_totals(&document, &mut findings);
    if let Some(changed) = changed {
        analyze_changed(root, &document, changed, &mut findings);
    }
    Ok(findings)
}

fn analyze_totals(document: &Value, findings: &mut Vec<Finding>) {
    let Some(totals) = document.pointer("/data/0/totals") else {
        return;
    };
    for kind in ["lines", "branches"] {
        let Some(percent) = totals
            .get(kind)
            .and_then(|summary| summary.get("percent"))
            .and_then(Value::as_f64)
        else {
            continue;
        };
        if percent + f64::EPSILON >= REQUIRED_PERCENT {
            continue;
        }
        let mut metrics = BTreeMap::new();
        metrics.insert("actual_percent".into(), percent);
        metrics.insert("required_percent".into(), REQUIRED_PERCENT);
        findings.push(Finding::new(
            "coverage_floor",
            Severity::Deny,
            Confidence::Proven,
            SourceSpan::file("Cargo.toml".into()),
            Evidence {
                summary: format!("workspace {kind} coverage is {percent:.2}%, below 95%"),
                details: vec![
                    "Measured by cargo-llvm-cov for the requested workspace targets.".into(),
                ],
                metrics,
            },
            kind,
        ));
    }
}

fn analyze_changed(
    root: &Path,
    document: &Value,
    changed: &BTreeMap<PathBuf, BTreeSet<usize>>,
    findings: &mut Vec<Finding>,
) {
    let Some(files) = document.pointer("/data/0/files").and_then(Value::as_array) else {
        return;
    };
    let mut observed = BTreeSet::new();
    for file in files {
        let Some(filename) = file.get("filename").and_then(Value::as_str) else {
            continue;
        };
        let path = repository_path(root, Path::new(filename));
        let Some(lines) = changed.get(&path) else {
            continue;
        };
        let Some(segments) = file.get("segments").and_then(Value::as_array) else {
            continue;
        };
        for segment in segments {
            let Some(values) = segment.as_array() else {
                continue;
            };
            let Some(line) = values
                .first()
                .and_then(Value::as_u64)
                .map(|line| line as usize)
            else {
                continue;
            };
            let count = values.get(2).and_then(Value::as_u64).unwrap_or(0);
            let has_count = values.get(3).and_then(Value::as_bool).unwrap_or(false);
            let gap = values.get(5).and_then(Value::as_bool).unwrap_or(false);
            if !lines.contains(&line) || !has_count || gap || !observed.insert((path.clone(), line))
            {
                continue;
            }
            if count == 0 {
                findings.push(uncovered(&path, line, "line"));
            }
        }
        if let Some(branches) = file.get("branches").and_then(Value::as_array) {
            for branch in branches {
                let Some(values) = branch.as_array() else {
                    continue;
                };
                let Some(line) = values
                    .first()
                    .and_then(Value::as_u64)
                    .map(|line| line as usize)
                else {
                    continue;
                };
                if !lines.contains(&line) {
                    continue;
                }
                let true_count = values.get(4).and_then(Value::as_u64).unwrap_or(0);
                let false_count = values.get(5).and_then(Value::as_u64).unwrap_or(0);
                if true_count == 0 || false_count == 0 {
                    findings.push(uncovered(&path, line, "branch"));
                }
            }
        }
    }
}

fn uncovered(path: &Path, line: usize, kind: &str) -> Finding {
    Finding::new(
        "changed_code_uncovered",
        Severity::Deny,
        Confidence::Proven,
        SourceSpan {
            path: path.to_path_buf(),
            line,
            column: 1,
        },
        Evidence {
            summary: format!("changed executable {kind} is not covered"),
            details: vec!["New critical behavior must be reached by the review test suite.".into()],
            metrics: BTreeMap::new(),
        },
        &format!("{kind}:{line}"),
    )
    .with_remediation(
        "Add a behavioral test",
        "Exercise the changed path and assert its observable result or diagnostic.",
    )
}

fn repository_path(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_totals_are_hard_findings() {
        let document: Value = serde_json::from_str(
            r#"{"data":[{"totals":{"lines":{"percent":94.9},"branches":{"percent":99.0}}}]}"#,
        )
        .unwrap();
        let mut findings = Vec::new();
        analyze_totals(&document, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].rule, "coverage_floor");
    }

    #[test]
    fn uncovered_changed_segment_is_rejected() {
        let document: Value = serde_json::from_str(
            r#"{"data":[{"files":[{"filename":"/repo/a.rs","segments":[[7,1,0,true,true,false]]}]}]}"#,
        )
        .unwrap();
        let changed = BTreeMap::from([(PathBuf::from("a.rs"), BTreeSet::from([7]))]);
        let mut findings = Vec::new();
        analyze_changed(Path::new("/repo"), &document, &changed, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].rule, "changed_code_uncovered");
    }
}
