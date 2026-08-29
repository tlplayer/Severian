use crate::model::{Finding, Severity};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn load(path: &Path) -> Result<BTreeSet<String>, String> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let source = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let mut fingerprints = BTreeSet::new();
    let mut current = Vec::new();
    for line in source.lines() {
        if line.trim() == "[[finding]]" {
            validate_entry(path, &current)?;
            current.clear();
        } else if !line.trim().is_empty() && !line.trim_start().starts_with('#') {
            current.push(line.trim().to_string());
            if let Some(value) = value(line, "fingerprint") {
                fingerprints.insert(value);
            }
        }
    }
    validate_entry(path, &current)?;
    Ok(fingerprints)
}

fn validate_entry(path: &Path, lines: &[String]) -> Result<(), String> {
    if lines.is_empty() {
        return Ok(());
    }
    for field in ["fingerprint", "owner", "reason", "issue", "expires"] {
        if !lines.iter().any(|line| value(line, field).is_some()) {
            return Err(format!(
                "{}: baseline finding is missing `{field}`",
                path.display()
            ));
        }
    }
    let expires = lines
        .iter()
        .find_map(|line| value(line, "expires"))
        .expect("validated baseline expiry exists");
    if !valid_date(&expires) {
        return Err(format!(
            "{}: baseline expiry `{expires}` must use YYYY-MM-DD",
            path.display()
        ));
    }
    if expires < today() {
        return Err(format!(
            "{}: baseline finding expired on {expires}",
            path.display()
        ));
    }
    Ok(())
}

fn valid_date(value: &str) -> bool {
    value.len() == 10
        && value.as_bytes()[4] == b'-'
        && value.as_bytes()[7] == b'-'
        && value
            .chars()
            .enumerate()
            .all(|(index, character)| matches!(index, 4 | 7) || character.is_ascii_digit())
}

fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 86_400;
    civil_date(days as i64)
}

fn civil_date(days_since_epoch: i64) -> String {
    let days = days_since_epoch + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

fn value(line: &str, field: &str) -> Option<String> {
    let suffix = line
        .strip_prefix(field)?
        .trim_start()
        .strip_prefix('=')?
        .trim();
    Some(suffix.strip_prefix('"')?.strip_suffix('"')?.to_string())
}

pub fn apply(findings: &mut [Finding], fingerprints: &BTreeSet<String>) {
    for finding in findings {
        finding.baseline = fingerprints.contains(&finding.fingerprint);
    }
}

pub fn write(path: &Path, findings: &[Finding]) -> Result<(), String> {
    let mut output = String::from(
        "# Existing health debt. Every entry needs an owner, reason, issue, and expiry.\n\n",
    );
    let mut fingerprints = findings
        .iter()
        .filter(|finding| finding.severity == Severity::Deny)
        .map(|finding| finding.fingerprint.as_str())
        .collect::<Vec<_>>();
    fingerprints.sort_unstable();
    fingerprints.dedup();
    for fingerprint in fingerprints {
        output.push_str(&format!(
            "[[finding]]\nfingerprint = \"{fingerprint}\"\nowner = \"compiler\"\nreason = \"Existing hard debt at initial code-health baseline\"\nissue = \"HEALTH-BASELINE\"\nexpires = \"2026-11-30\"\n\n"
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    fs::write(path, output).map_err(|error| format!("could not write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_converts_to_1970_01_01() {
        assert_eq!(civil_date(0), "1970-01-01");
    }

    #[test]
    fn invalid_expiry_is_rejected() {
        assert!(validate_entry(
            Path::new("baseline.toml"),
            &[
                "fingerprint = \"x\"".into(),
                "owner = \"compiler\"".into(),
                "reason = \"debt\"".into(),
                "issue = \"HEALTH-1\"".into(),
                "expires = \"1970-01-01\"".into(),
            ]
        )
        .is_err());
    }
}
