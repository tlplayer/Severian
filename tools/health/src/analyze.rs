use crate::architecture::{
    cargo_manifests, is_artifact_path, is_user_input_path, package_name, parse_architecture_allow,
    severian_dependencies,
};
use crate::model::{Confidence, Evidence, Finding, Severity, SourceSpan};
use crate::repository;
use crate::source::{
    self, call_names, contains_code, extract_enums, extract_functions, maximum_nesting,
    normalize_exact, normalize_renamed, normalized_line, FunctionBody, SourceUnit,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn run(root: &Path, changed: Option<&BTreeSet<PathBuf>>) -> Result<Vec<Finding>, String> {
    let trace = std::env::var_os("SEVERIAN_HEALTH_TRACE").is_some();
    let mut phase = Instant::now();
    let units = source::load(root)?;
    trace_phase(trace, "load", &mut phase);
    let mut findings = Vec::new();
    analyze_source_files(&units, &mut findings);
    trace_phase(trace, "source rules", &mut phase);
    analyze_architecture(root, &mut findings)?;
    analyze_bootstrap_mirror(root, &mut findings)?;
    trace_phase(trace, "architecture", &mut phase);
    analyze_clones(&units, &mut findings);
    trace_phase(trace, "clones", &mut phase);
    analyze_parallel_catalogs(&units, &mut findings);
    trace_phase(trace, "catalogs", &mut phase);
    findings.extend(crate::graph::analyze(&units));
    trace_phase(trace, "symbol graph", &mut phase);
    analyze_risk(root, &units, &mut findings);
    trace_phase(trace, "risk", &mut phase);
    if let Some(changed) = changed {
        analyze_vertical_tests(changed, &mut findings);
        findings.retain(|finding| {
            changed.contains(&finding.primary_span.path)
                || finding
                    .related_spans
                    .iter()
                    .any(|span| changed.contains(&span.path))
        });
    }
    findings.sort_by(|left, right| {
        left.severity
            .cmp(&right.severity)
            .then(left.confidence.cmp(&right.confidence))
            .then(left.primary_span.path.cmp(&right.primary_span.path))
            .then(left.primary_span.line.cmp(&right.primary_span.line))
            .then(left.rule.cmp(&right.rule))
    });
    Ok(findings)
}

fn analyze_bootstrap_mirror(root: &Path, findings: &mut Vec<Finding>) -> Result<(), String> {
    let bridge_path = PathBuf::from("library/core/compile/interop/rust/lib.rs");
    let source_path = PathBuf::from("library/core/compile/src/mod.sev");
    let bridge = fs::read_to_string(root.join(&bridge_path))
        .map_err(|error| format!("could not read {}: {error}", bridge_path.display()))?;
    if !root.join(&source_path).is_file()
        || !bridge.contains("path: \"src/mod.sev\"")
        || !bridge.contains("include_str!(\"../../src/mod.sev\")")
    {
        findings.push(Finding::new(
            "bootstrap_semantic_drift",
            Severity::Deny,
            Confidence::Proven,
            SourceSpan::file(bridge_path),
            Evidence {
                summary:
                    "Rust bootstrap bridge does not embed the canonical Severian protocol source"
                        .into(),
                details: vec![format!("canonical source: {}", source_path.display())],
                metrics: BTreeMap::new(),
            },
            "core-compile-source-embedding",
        ));
    }
    Ok(())
}

fn analyze_vertical_tests(changed: &BTreeSet<PathBuf>, findings: &mut Vec<Finding>) {
    let semantic_change = changed.iter().any(|path| {
        path.starts_with("compiler/universal")
            || path.starts_with("compiler/frontend")
            || path.starts_with("compiler/transforms")
    });
    let vertical_test = changed.iter().any(|path| {
        path.extension().and_then(|value| value.to_str()) == Some("sev")
            || path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name == "tests.rs" || name.ends_with("_test.rs"))
            || path
                .components()
                .any(|component| component.as_os_str() == "tests")
    });
    if semantic_change && !vertical_test {
        let path = changed
            .iter()
            .find(|path| path.starts_with("compiler"))
            .cloned()
            .unwrap_or_else(|| PathBuf::from("compiler"));
        findings.push(
            Finding::new(
                "vertical_test_missing",
                Severity::Warning,
                Confidence::High,
                SourceSpan::file(path),
                Evidence {
                    summary: "compiler semantics changed without a source-level vertical test"
                        .into(),
                    details: vec![
                        "Changed paths contain no .sev example or Rust test module.".into()
                    ],
                    metrics: BTreeMap::new(),
                },
                "semantic-change-without-test",
            )
            .with_remediation(
                "Add a source-to-result test",
                "Cover the concept from Severian source through artifact or stable diagnostic.",
            ),
        );
    }
}

fn trace_phase(enabled: bool, name: &str, started: &mut Instant) {
    if enabled {
        eprintln!(
            "health trace: {name} {:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    *started = Instant::now();
}

fn analyze_source_files(units: &[SourceUnit], findings: &mut Vec<Finding>) {
    for unit in units {
        if unit.lines > 800 {
            let mut metrics = BTreeMap::new();
            metrics.insert("lines".into(), unit.lines as f64);
            metrics.insert("limit".into(), 800.0);
            findings.push(
                Finding::new(
                    "source_file_limit",
                    Severity::Deny,
                    Confidence::Proven,
                    SourceSpan::file(unit.path.clone()),
                    Evidence {
                        summary: format!(
                            "{} has {} lines, exceeding the repository limit of 800",
                            unit.path.display(),
                            unit.lines
                        ),
                        details: vec![
                            "Size is a hard invariant; cohesion is analyzed separately.".into()
                        ],
                        metrics,
                    },
                    "over-800-lines",
                )
                .with_remediation(
                    "Extract a coherent module",
                    "Split only at a demonstrated responsibility or dependency boundary.",
                ),
            );
        }
        analyze_unsafe(unit, findings);
        analyze_user_input_panics(unit, findings);
        analyze_string_dispatch(unit, findings);
        analyze_unverified_transforms(unit, findings);
        analyze_nondeterminism(unit, findings);
    }
}

fn analyze_unsafe(unit: &SourceUnit, findings: &mut Vec<Finding>) {
    let lines = unit.text.lines().collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        if !contains_code(line, "unsafe {") && !line.trim_start().starts_with("unsafe fn ") {
            continue;
        }
        if (index.saturating_sub(3)..index).any(|line| lines[line].contains("SAFETY:")) {
            continue;
        }
        findings.push(
            Finding::new(
                "unsafe_without_contract",
                Severity::Deny,
                Confidence::Proven,
                SourceSpan {
                    path: unit.path.clone(),
                    line: index + 1,
                    column: line.find("unsafe").unwrap_or(0) + 1,
                },
                Evidence {
                    summary: "unsafe code has no adjacent SAFETY contract".into(),
                    details: vec![
                        "The preceding three lines contain no safety explanation.".into(),
                    ],
                    metrics: BTreeMap::new(),
                },
                &format!("unsafe:{}", normalized_line(line)),
            )
            .with_remediation(
                "Document or remove unsafe",
                "State caller obligations and why every unsafe precondition holds.",
            ),
        );
    }
}

fn analyze_user_input_panics(unit: &SourceUnit, findings: &mut Vec<Finding>) {
    if !is_user_input_path(&unit.path) {
        return;
    }
    for (index, line) in unit.text.lines().enumerate() {
        let operation = [".unwrap()", ".expect(", "panic!(", "unreachable!("]
            .into_iter()
            .find(|operation| contains_code(line, operation));
        let Some(operation) = operation else { continue };
        if line.contains("bootstrap defines") || line.contains("internal invariant") {
            continue;
        }
        findings.push(
            Finding::new(
                "user_input_panic",
                Severity::Warning,
                Confidence::High,
                SourceSpan {
                    path: unit.path.clone(),
                    line: index + 1,
                    column: line.find(operation).unwrap_or(0) + 1,
                },
                Evidence {
                    summary: format!("source-input compiler path reaches {operation}"),
                    details: vec![
                        "This call-graph seed requires review; proven invariants need a reason."
                            .into(),
                    ],
                    metrics: BTreeMap::new(),
                },
                &format!("{operation}:{}", normalized_line(line)),
            )
            .with_remediation(
                "Return a diagnostic",
                "Convert user-controlled failure into a stable diagnostic and source span.",
            ),
        );
    }
}

fn analyze_string_dispatch(unit: &SourceUnit, findings: &mut Vec<Finding>) {
    if !unit.path.starts_with("compiler") {
        return;
    }
    for (index, line) in unit.text.lines().enumerate() {
        let lower = line.to_ascii_lowercase();
        let semantic_word = [
            "operator",
            "primitive",
            "dtype",
            "compile",
            "backend",
            "tensor",
            "route",
            "effect",
        ]
        .into_iter()
        .any(|word| lower.contains(word));
        let string_decision =
            (line.contains(".as_str()") || line.contains("== \"") || line.contains("match "))
                && line.contains('"');
        if semantic_word && string_decision {
            findings.push(
                Finding::new(
                    "stringly_semantic_dispatch",
                    Severity::Warning,
                    Confidence::Heuristic,
                    SourceSpan {
                        path: unit.path.clone(),
                        line: index + 1,
                        column: 1,
                    },
                    Evidence {
                        summary: "compiler semantic decision appears to use a string spelling"
                            .into(),
                        details: vec![line.trim().to_string()],
                        metrics: BTreeMap::new(),
                    },
                    &normalized_line(line),
                )
                .with_remediation(
                    "Use the canonical semantic ID",
                    "Keep strings at parser/bootstrap boundaries and pass structural IDs afterward.",
                ),
            );
        }
    }
}

fn analyze_unverified_transforms(unit: &SourceUnit, findings: &mut Vec<Finding>) {
    if !unit.path.starts_with("compiler/transforms") {
        return;
    }
    let count = ["fn lower", "fn transform", "fn optimize", "impl Pass"]
        .into_iter()
        .map(|needle| unit.text.matches(needle).count())
        .sum::<usize>();
    if count > 0 && !unit.text.contains("verify(") && !unit.text.contains("PassContract") {
        let mut metrics = BTreeMap::new();
        metrics.insert("transform_entry_points".into(), count as f64);
        findings.push(
            Finding::new(
                "unverified_transform",
                Severity::Warning,
                Confidence::High,
                SourceSpan::file(unit.path.clone()),
                Evidence {
                    summary: "transform entry points have no verifier or pass contract".into(),
                    details: vec![
                        "A caller may verify this transform; that call edge should be explicit."
                            .into(),
                    ],
                    metrics,
                },
                "transform-without-contract",
            )
            .with_remediation(
                "Declare and enforce a pass contract",
                "Verify requirements before and established or preserved invariants after.",
            ),
        );
    }
}

fn analyze_nondeterminism(unit: &SourceUnit, findings: &mut Vec<Finding>) {
    if !unit.path.starts_with("compiler") {
        return;
    }
    for (index, line) in unit.text.lines().enumerate() {
        let source = if line.contains("SystemTime::now") {
            Some("wall-clock time")
        } else if line.contains("HashMap::") && is_artifact_path(&unit.path) {
            Some("unordered map")
        } else {
            None
        };
        let Some(source) = source else { continue };
        findings.push(Finding::new(
            "nondeterministic_artifact",
            Severity::Warning,
            Confidence::Heuristic,
            SourceSpan {
                path: unit.path.clone(),
                line: index + 1,
                column: 1,
            },
            Evidence {
                summary: format!("{source} is used in an artifact or package path"),
                details: vec![line.trim().to_string()],
                metrics: BTreeMap::new(),
            },
            &format!("{source}:{}", normalized_line(line)),
        ));
    }
}

fn analyze_architecture(root: &Path, findings: &mut Vec<Finding>) -> Result<(), String> {
    let root_manifest = fs::read_to_string(root.join("Cargo.toml"))
        .map_err(|error| format!("could not read Cargo.toml: {error}"))?;
    let allowed = parse_architecture_allow(&root_manifest);
    for manifest in cargo_manifests(root)? {
        let relative = manifest
            .strip_prefix(root)
            .unwrap_or(&manifest)
            .to_path_buf();
        if relative.starts_with("tools") {
            continue;
        }
        let source = fs::read_to_string(&manifest)
            .map_err(|error| format!("could not read {}: {error}", manifest.display()))?;
        let Some(package) = package_name(&source) else {
            continue;
        };
        let dependencies = severian_dependencies(&source);
        let declared = allowed.get(&package).cloned().unwrap_or_default();
        for dependency in dependencies.difference(&declared) {
            findings.push(
                Finding::new(
                    "forbidden_dependency",
                    Severity::Deny,
                    Confidence::Proven,
                    SourceSpan::file(relative.clone()),
                    Evidence {
                        summary: format!(
                            "{package} depends on forbidden layer package {dependency}"
                        ),
                        details: vec![
                            "Dependency is absent from workspace architecture allowlist.".into(),
                        ],
                        metrics: BTreeMap::new(),
                    },
                    &format!("{package}->{dependency}"),
                )
                .with_remediation(
                    "Use an allowed boundary",
                    "Move the dependency behind its owning compiler layer.",
                ),
            );
        }
        if relative.starts_with("compiler/frontend") {
            for dependency in dependencies.iter().filter(|dependency| {
                dependency.contains("target")
                    || dependency.contains("backend")
                    || dependency.contains("mlir")
                    || dependency.contains("artifact")
            }) {
                findings.push(Finding::new(
                    "frontend_target_leak",
                    Severity::Deny,
                    Confidence::Proven,
                    SourceSpan::file(relative.clone()),
                    Evidence {
                        summary: format!(
                            "frontend package {package} directly depends on {dependency}"
                        ),
                        details: Vec::new(),
                        metrics: BTreeMap::new(),
                    },
                    &format!("{package}->{dependency}"),
                ));
            }
        }
    }
    Ok(())
}

fn analyze_clones(units: &[SourceUnit], findings: &mut Vec<Finding>) {
    let functions = units.iter().flat_map(extract_functions).collect::<Vec<_>>();
    let mut exact = BTreeMap::<String, Vec<&FunctionBody>>::new();
    let mut renamed = BTreeMap::<String, Vec<&FunctionBody>>::new();
    for function in &functions {
        if function.source.len() < 120 {
            continue;
        }
        exact
            .entry(normalize_exact(&function.source))
            .or_default()
            .push(function);
        renamed
            .entry(normalize_renamed(&function.source))
            .or_default()
            .push(function);
    }
    for group in exact.values().filter(|group| group.len() >= 2) {
        clone_finding("exact_clone", Confidence::Proven, group, findings);
    }
    for group in renamed.values().filter(|group| group.len() >= 3) {
        let forms = group
            .iter()
            .map(|function| normalize_exact(&function.source))
            .collect::<BTreeSet<_>>();
        if forms.len() > 1 {
            clone_finding("renamed_clone", Confidence::High, group, findings);
        }
    }
}

fn clone_finding(
    rule: &str,
    confidence: Confidence,
    group: &[&FunctionBody],
    findings: &mut Vec<Finding>,
) {
    let primary = group[0];
    let names = group
        .iter()
        .map(|function| format!("{}::{}", function.path.display(), function.name))
        .collect::<Vec<_>>();
    let mut metrics = BTreeMap::new();
    metrics.insert("instances".into(), group.len() as f64);
    metrics.insert(
        "normalized_bytes".into(),
        normalize_exact(&primary.source).len() as f64,
    );
    findings.push(
        Finding::new(
            rule,
            Severity::Warning,
            confidence,
            SourceSpan {
                path: primary.path.clone(),
                line: primary.line,
                column: 1,
            },
            Evidence {
                summary: format!(
                    "{} functions share the same normalized implementation",
                    group.len()
                ),
                details: names.clone(),
                metrics,
            },
            &names.join("|"),
        )
        .with_related(
            group[1..]
                .iter()
                .map(|function| SourceSpan {
                    path: function.path.clone(),
                    line: function.line,
                    column: 1,
                })
                .collect(),
        )
        .with_remediation(
            "Classify the variation",
            "Use a table, generic, strategy, or thin wrappers only when evidence supports it.",
        ),
    );
}

fn analyze_parallel_catalogs(units: &[SourceUnit], findings: &mut Vec<Finding>) {
    let catalogs = units.iter().flat_map(extract_enums).collect::<Vec<_>>();
    for left_index in 0..catalogs.len() {
        for right in &catalogs[left_index + 1..] {
            let left = &catalogs[left_index];
            if left.variants.len() < 4 || right.variants.len() < 4 || left.path == right.path {
                continue;
            }
            let intersection = left.variants.intersection(&right.variants).count();
            let union = left.variants.union(&right.variants).count();
            let similarity = intersection as f64 / union as f64;
            if similarity < 0.75 {
                continue;
            }
            let mut metrics = BTreeMap::new();
            metrics.insert("jaccard".into(), similarity);
            metrics.insert("shared_variants".into(), intersection as f64);
            let identity = format!(
                "{}::{}|{}::{}",
                left.path.display(),
                left.name,
                right.path.display(),
                right.name
            );
            findings.push(
                Finding::new(
                    "parallel_semantic_catalog",
                    Severity::Warning,
                    Confidence::High,
                    SourceSpan {
                        path: left.path.clone(),
                        line: left.line,
                        column: 1,
                    },
                    Evidence {
                        summary: format!(
                            "{} and {} share {:.0}% of variant names",
                            left.name,
                            right.name,
                            similarity * 100.0
                        ),
                        details: vec![format!(
                            "shared: {}",
                            left.variants
                                .intersection(&right.variants)
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        )],
                        metrics,
                    },
                    &identity,
                )
                .with_related(vec![SourceSpan {
                    path: right.path.clone(),
                    line: right.line,
                    column: 1,
                }])
                .with_remediation(
                    "Identify the semantic owner",
                    "Keep representations distinct but project shared semantics from one owner.",
                ),
            );
        }
    }
}

fn analyze_risk(root: &Path, units: &[SourceUnit], findings: &mut Vec<Finding>) {
    let churned = repository::churn_90d(root);
    let mut metrics = units
        .iter()
        .map(|unit| {
            let nesting = maximum_nesting(&unit.text);
            let branches = unit.text.matches("if ").count()
                + unit.text.matches("match ").count()
                + unit.text.matches("=>").count();
            let panics = unit.text.matches(".unwrap()").count()
                + unit.text.matches(".expect(").count()
                + unit.text.matches("panic!(").count();
            let fanout = call_names(&unit.text).len();
            let churn = usize::from(churned.contains(&unit.path));
            let raw = unit.lines as f64 / 800.0
                + nesting as f64 / 12.0
                + branches as f64 / 150.0
                + panics as f64 / 20.0
                + fanout as f64 / 100.0;
            let score = (1.0 + (1.0 + churn as f64).ln()) * raw;
            (unit, nesting, branches, panics, fanout, churn, score)
        })
        .collect::<Vec<_>>();
    metrics.sort_by(|left, right| left.6.total_cmp(&right.6));
    let threshold = metrics
        .get(metrics.len().saturating_mul(9) / 10)
        .map_or(f64::INFINITY, |metric| metric.6);
    for (unit, nesting, branches, panics, fanout, churn, score) in metrics {
        if score < threshold || unit.lines < 200 {
            continue;
        }
        let mut values = BTreeMap::new();
        values.insert("risk_score".into(), score);
        values.insert("lines".into(), unit.lines as f64);
        values.insert("maximum_nesting".into(), nesting as f64);
        values.insert("branch_signals".into(), branches as f64);
        values.insert("panic_signals".into(), panics as f64);
        values.insert("fan_out".into(), fanout as f64);
        values.insert("changed_in_90d".into(), churn as f64);
        findings.push(Finding::new(
            "hotspot_risk",
            Severity::Information,
            Confidence::Trend,
            SourceSpan::file(unit.path.clone()),
            Evidence {
                summary: "file is in the repository top risk decile".into(),
                details: vec!["Risk orders cleanup work and never fails CI by itself.".into()],
                metrics: values,
            },
            "top-risk-decile",
        ));
    }
}
