use crate::model::{Finding, Severity};
use crate::options::Format;
use std::collections::BTreeMap;

pub fn render(findings: &[Finding], format: Format) -> String {
    match format {
        Format::Human => human(findings),
        Format::Json => json(findings),
        Format::Sarif => sarif(findings),
    }
}

fn human(findings: &[Finding]) -> String {
    let mut output = String::new();
    for finding in findings {
        let baseline = if finding.baseline { " [baseline]" } else { "" };
        output.push_str(&format!(
            "health[{}] {} {}{} at {}\n{}\n",
            finding.rule,
            finding.severity.name(),
            finding.confidence.name(),
            baseline,
            finding.primary_span,
            finding.evidence.summary
        ));
        for detail in &finding.evidence.details {
            output.push_str(&format!("  evidence: {detail}\n"));
        }
        for (name, value) in &finding.evidence.metrics {
            output.push_str(&format!("  {name}: {value:.3}\n"));
        }
        for span in &finding.related_spans {
            output.push_str(&format!("  related: {span}\n"));
        }
        for remediation in &finding.remediation {
            output.push_str(&format!(
                "  remediation: {} — {}\n",
                remediation.title, remediation.rationale
            ));
        }
        output.push_str(&format!("  fingerprint: {}\n\n", finding.fingerprint));
    }
    let counts = counts(findings);
    let new_failures = findings
        .iter()
        .filter(|finding| !finding.baseline && finding.severity == Severity::Deny)
        .count();
    output.push_str(&format!(
        "health summary: {} findings ({} deny, {} warning, {} information), {} baselined, {} new hard failures\n",
        findings.len(),
        counts.get("deny").copied().unwrap_or(0),
        counts.get("warning").copied().unwrap_or(0),
        counts.get("information").copied().unwrap_or(0),
        findings.iter().filter(|finding| finding.baseline).count(),
        new_failures,
    ));
    output
}

fn json(findings: &[Finding]) -> String {
    let entries = findings
        .iter()
        .map(|finding| {
            let details = finding
                .evidence
                .details
                .iter()
                .map(|detail| format!("\"{}\"", escape(detail)))
                .collect::<Vec<_>>()
                .join(",");
            let related = finding
                .related_spans
                .iter()
                .map(|span| {
                    format!(
                        "{{\"path\":\"{}\",\"line\":{},\"column\":{}}}",
                        escape(&span.path.display().to_string()),
                        span.line,
                        span.column
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let metrics = finding
                .evidence
                .metrics
                .iter()
                .map(|(name, value)| format!("\"{}\":{}", escape(name), value))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"rule\":\"{}\",\"severity\":\"{}\",\"confidence\":\"{}\",\"baseline\":{},\"fingerprint\":\"{}\",\"primary_span\":{{\"path\":\"{}\",\"line\":{},\"column\":{}}},\"related_spans\":[{}],\"evidence\":{{\"summary\":\"{}\",\"details\":[{}],\"metrics\":{{{}}}}}}}",
                escape(&finding.rule),
                finding.severity.name(),
                finding.confidence.name(),
                finding.baseline,
                escape(&finding.fingerprint),
                escape(&finding.primary_span.path.display().to_string()),
                finding.primary_span.line,
                finding.primary_span.column,
                related,
                escape(&finding.evidence.summary),
                details,
                metrics,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"version\":1,\"findings\":[{entries}]}}\n")
}

fn sarif(findings: &[Finding]) -> String {
    let mut rules = BTreeMap::<&str, (&str, &str)>::new();
    for finding in findings {
        rules
            .entry(&finding.rule)
            .or_insert((finding.severity.name(), finding.evidence.summary.as_str()));
    }
    let rules = rules
        .into_iter()
        .map(|(id, (severity, summary))| {
            format!(
                "{{\"id\":\"{}\",\"shortDescription\":{{\"text\":\"{}\"}},\"defaultConfiguration\":{{\"level\":\"{}\"}}}}",
                escape(id),
                escape(summary),
                sarif_level(severity)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let results = findings
        .iter()
        .map(|finding| {
            format!(
                "{{\"ruleId\":\"{}\",\"level\":\"{}\",\"message\":{{\"text\":\"{} [{} confidence; fingerprint {}]\"}},\"locations\":[{{\"physicalLocation\":{{\"artifactLocation\":{{\"uri\":\"{}\"}},\"region\":{{\"startLine\":{},\"startColumn\":{}}}}}}}],\"baselineState\":\"{}\"}}",
                escape(&finding.rule),
                sarif_level(finding.severity.name()),
                escape(&finding.evidence.summary),
                finding.confidence.name(),
                escape(&finding.fingerprint),
                escape(&finding.primary_span.path.display().to_string()),
                finding.primary_span.line,
                finding.primary_span.column,
                if finding.baseline { "unchanged" } else { "new" },
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"version\":\"2.1.0\",\"$schema\":\"https://json.schemastore.org/sarif-2.1.0.json\",\"runs\":[{{\"tool\":{{\"driver\":{{\"name\":\"Severian Health\",\"rules\":[{rules}]}}}},\"results\":[{results}]}}]}}\n"
    )
}

fn counts(findings: &[Finding]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for finding in findings {
        *counts.entry(finding.severity.name()).or_default() += 1;
    }
    counts
}

fn sarif_level(severity: &str) -> &'static str {
    match severity {
        "deny" => "error",
        "warning" => "warning",
        _ => "note",
    }
}

fn escape(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Confidence, Evidence, SourceSpan};

    #[test]
    fn structured_reports_escape_source_text() {
        let finding = Finding::new(
            "quoted",
            Severity::Warning,
            Confidence::High,
            SourceSpan::file("a.rs".into()),
            Evidence {
                summary: "a \"quote\"".into(),
                ..Evidence::default()
            },
            "quote",
        );
        let json = json(&[finding.clone()]);
        let sarif = sarif(&[finding]);
        assert!(json.contains("a \\\"quote\\\""));
        assert!(sarif.contains("a \\\"quote\\\""));
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());
        assert!(serde_json::from_str::<serde_json::Value>(&sarif).is_ok());
    }
}
