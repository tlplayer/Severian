use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    Deny,
    Warning,
    Information,
}

impl Severity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deny => "deny",
            Self::Warning => "warning",
            Self::Information => "information",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Confidence {
    Proven,
    High,
    Heuristic,
    Trend,
}

impl Confidence {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Proven => "proven",
            Self::High => "high",
            Self::Heuristic => "heuristic",
            Self::Trend => "trend",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    pub path: PathBuf,
    pub line: usize,
    pub column: usize,
}

impl SourceSpan {
    pub fn file(path: PathBuf) -> Self {
        Self {
            path,
            line: 1,
            column: 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evidence {
    pub summary: String,
    pub details: Vec<String>,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Remediation {
    pub title: String,
    pub rationale: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub rule: String,
    pub severity: Severity,
    pub confidence: Confidence,
    pub primary_span: SourceSpan,
    pub related_spans: Vec<SourceSpan>,
    pub evidence: Evidence,
    pub remediation: Vec<Remediation>,
    pub fingerprint: String,
    pub baseline: bool,
}

impl Finding {
    pub fn new(
        rule: impl Into<String>,
        severity: Severity,
        confidence: Confidence,
        primary_span: SourceSpan,
        evidence: Evidence,
        identity: &str,
    ) -> Self {
        let rule = rule.into();
        let fingerprint = fingerprint(&rule, &primary_span.path, identity);
        Self {
            rule,
            severity,
            confidence,
            primary_span,
            related_spans: Vec::new(),
            evidence,
            remediation: Vec::new(),
            fingerprint,
            baseline: false,
        }
    }

    pub fn with_related(mut self, spans: Vec<SourceSpan>) -> Self {
        self.related_spans = spans;
        self
    }

    pub fn with_remediation(mut self, title: &str, rationale: &str) -> Self {
        self.remediation.push(Remediation {
            title: title.into(),
            rationale: rationale.into(),
        });
        self
    }

    pub const fn fails_gate(&self, deny_warnings: bool) -> bool {
        if self.baseline {
            return false;
        }
        matches!(self.severity, Severity::Deny)
            || (deny_warnings && matches!(self.severity, Severity::Warning))
    }
}

fn fingerprint(rule: &str, path: &std::path::Path, identity: &str) -> String {
    let input = format!("{rule}\0{}\0{identity}", path.display());
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in input.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{rule}:{hash:016x}")
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}:{}",
            self.path.display(),
            self.line,
            self.column
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_do_not_depend_on_line_numbers() {
        let first = Finding::new(
            "rule",
            Severity::Warning,
            Confidence::High,
            SourceSpan {
                path: "compiler/file.rs".into(),
                line: 4,
                column: 1,
            },
            Evidence::default(),
            "symbol",
        );
        let second = Finding::new(
            "rule",
            Severity::Warning,
            Confidence::High,
            SourceSpan {
                path: "compiler/file.rs".into(),
                line: 90,
                column: 8,
            },
            Evidence::default(),
            "symbol",
        );
        assert_eq!(first.fingerprint, second.fingerprint);
    }
}
