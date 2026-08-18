use severian_package::{BuildPolicy, FileLimitException};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

mod dependency;

pub use dependency::{
    analyze_dependencies, ArchitectureDependency, ArchitectureFinding, ArchitectureNode,
    DependencyAnalysis, DependencyStat,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileBudgetFinding {
    pub severity: &'static str,
    pub path: PathBuf,
    pub lines: usize,
    pub limit: usize,
    pub exception_reason: Option<String>,
    pub message: String,
}

pub fn check_file_budgets(policy: &BuildPolicy) -> Result<Vec<FileBudgetFinding>, String> {
    let mut files = Vec::new();
    collect_files(&policy.root, &policy.root, &mut files).map_err(|error| error.to_string())?;
    files.sort();
    let today = current_date();
    let mut findings = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(&policy.root)
            .unwrap_or(&path)
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if !policy.files.includes(&relative) {
            continue;
        }
        let source = fs::read_to_string(&path)
            .map_err(|error| format!("could not inspect {}: {error}", path.display()))?;
        let lines = source.lines().count();
        let (soft, hard, exception) = policy.files.limits_for(&relative);
        if let Some(exception) = exception {
            if exception
                .expires
                .as_deref()
                .is_some_and(|expiry| expiry < today.as_str())
            {
                findings.push(expired_exception(&path, lines, exception));
                continue;
            }
        }
        if lines > hard {
            findings.push(FileBudgetFinding {
                severity: "error",
                path,
                lines,
                limit: hard,
                exception_reason: exception.map(|exception| exception.reason.clone()),
                message: format!(
                    "file has {lines} lines; the hard architectural limit is {hard}. {}",
                    split_suggestion(&relative)
                ),
            });
        } else if lines > soft {
            findings.push(FileBudgetFinding {
                severity: "warning",
                path,
                lines,
                limit: soft,
                exception_reason: exception.map(|exception| exception.reason.clone()),
                message: format!(
                    "file has {lines} lines; the soft architectural limit is {soft}. {}",
                    split_suggestion(&relative)
                ),
            });
        }
    }
    Ok(findings)
}

fn collect_files(root: &Path, directory: &Path, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|name| name.to_str());
            if matches!(name, Some("target" | ".git" | ".codex" | ".agents")) {
                continue;
            }
            collect_files(root, &path, output)?;
        } else if path.is_file() && path.starts_with(root) {
            output.push(path);
        }
    }
    Ok(())
}

fn expired_exception(
    path: &Path,
    lines: usize,
    exception: &FileLimitException,
) -> FileBudgetFinding {
    FileBudgetFinding {
        severity: "error",
        path: path.to_path_buf(),
        lines,
        limit: exception.hard_lines,
        exception_reason: Some(exception.reason.clone()),
        message: format!(
            "architectural exception expired on {}; split the file or renew it in review",
            exception.expires.as_deref().unwrap_or("an unknown date")
        ),
    }
}

fn split_suggestion(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("module");
    format!(
        "Split by responsibility, for example `{stem}/mod.rs`, `{stem}/types.rs`, and `{stem}/lower.rs`, or add a reviewed exception with a reason and expiry."
    )
}

fn current_date() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

// Howard Hinnant's civil-from-days conversion, with the epoch shifted from
// 1970-01-01 to the civil calendar origin.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
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
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_unix_epoch_to_calendar_date() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_000), (2024, 10, 4));
    }
}
