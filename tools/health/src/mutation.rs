use crate::model::{Confidence, Evidence, Finding, Severity, SourceSpan};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn analyze(root: &Path, report: &Path) -> Result<Vec<Finding>, String> {
    let source = fs::read_to_string(report).map_err(|error| {
        format!(
            "could not read mutation report {}: {error}",
            report.display()
        )
    })?;
    let document: Value = serde_json::from_str(&source)
        .map_err(|error| format!("invalid mutation JSON {}: {error}", report.display()))?;
    let mut survivors = Vec::new();
    collect_survivors(root, &document, &mut survivors);
    survivors.sort_by(|left, right| left.fingerprint.cmp(&right.fingerprint));
    survivors.dedup_by(|left, right| left.fingerprint == right.fingerprint);
    Ok(survivors)
}

fn collect_survivors(root: &Path, value: &Value, output: &mut Vec<Finding>) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_survivors(root, value, output);
            }
        }
        Value::Object(object) => {
            if outcome(object).is_some_and(|outcome| outcome.eq_ignore_ascii_case("survived")) {
                output.push(survivor(root, object));
                return;
            }
            for value in object.values() {
                collect_survivors(root, value, output);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn outcome(object: &Map<String, Value>) -> Option<&str> {
    ["outcome", "status", "result"]
        .into_iter()
        .find_map(|key| object.get(key).and_then(Value::as_str))
}

fn survivor(root: &Path, object: &Map<String, Value>) -> Finding {
    let path = find_string(object, &["file", "path", "filename"])
        .map(PathBuf::from)
        .map(|path| path.strip_prefix(root).unwrap_or(&path).to_path_buf())
        .unwrap_or_else(|| PathBuf::from("Cargo.toml"));
    let line = find_u64(object, &["line", "line_start", "start_line"]).unwrap_or(1) as usize;
    let description = find_string(object, &["description", "name", "mutant"])
        .unwrap_or("mutation changed behavior without failing a test");
    Finding::new(
        "mutation_survived",
        Severity::Deny,
        Confidence::Proven,
        SourceSpan {
            path,
            line,
            column: 1,
        },
        Evidence {
            summary: "a tested mutation survived".into(),
            details: vec![description.to_owned()],
            metrics: BTreeMap::new(),
        },
        description,
    )
    .with_remediation(
        "Strengthen the behavioral assertion",
        "Add a test that distinguishes the original compiler behavior from this mutation.",
    )
}

fn find_string<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(value) = object.get(*key).and_then(Value::as_str) {
            return Some(value);
        }
    }
    object
        .values()
        .filter_map(Value::as_object)
        .find_map(|nested| find_string(nested, keys))
}

fn find_u64(object: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    for key in keys {
        if let Some(value) = object.get(*key).and_then(Value::as_u64) {
            return Some(value);
        }
    }
    object
        .values()
        .filter_map(Value::as_object)
        .find_map(|nested| find_u64(nested, keys))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_nested_survivor_records() {
        let value: Value = serde_json::from_str(
            r#"{"outcomes":[{"outcome":"Survived","mutant":{"file":"a.rs","line":7,"description":"negate condition"}}]}"#,
        )
        .unwrap();
        let mut findings = Vec::new();
        collect_survivors(Path::new("/repo"), &value, &mut findings);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].primary_span.line, 7);
    }
}
