use crate::build_options::DiagnosticsMode;
use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitStatus,
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) const REPORT_ENV: &str = "SEVERIAN_RUNTIME_DIAGNOSTIC";
const HEADER: &str = "SEVERIAN_RUNTIME_DIAGNOSTIC_V1";

pub(super) fn parse_run_args(
    args: &[String],
) -> Result<(PathBuf, Vec<String>, Option<DiagnosticsMode>), String> {
    let separator = args.iter().position(|argument| argument == "--");
    let (target_arguments, application_arguments) = match separator {
        Some(index) => (&args[..index], args[index + 1..].to_vec()),
        None => (args, Vec::new()),
    };
    let mut input = None;
    let mut diagnostics = None;
    let mut index = 0;
    while index < target_arguments.len() {
        match target_arguments[index].as_str() {
            "--diagnostics" if index + 1 < target_arguments.len() => {
                diagnostics = Some(DiagnosticsMode::parse(&target_arguments[index + 1])?);
                index += 2;
            }
            value if value.starts_with("--diagnostics=") => {
                diagnostics = Some(DiagnosticsMode::parse(
                    value.trim_start_matches("--diagnostics="),
                )?);
                index += 1;
            }
            value if !value.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(value));
                index += 1;
            }
            _ => {
                return Err(
                    "run accepts one project or source path and an optional `--diagnostics` mode; put application arguments after `--`"
                        .into(),
                )
            }
        }
    }
    Ok((
        input.unwrap_or_else(|| PathBuf::from(".")),
        application_arguments,
        diagnostics,
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeDiagnostic {
    code: String,
    message: String,
    path: PathBuf,
    line: usize,
    column: usize,
    detail: String,
}

pub(super) fn report_path(binary: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let directory = binary.parent().unwrap_or_else(|| Path::new("."));
    let path = directory.join(format!(
        ".severian-runtime-{}-{nonce}.diagnostic",
        std::process::id()
    ));
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

pub(super) fn take_report(
    path: &Path,
    mode: DiagnosticsMode,
    binary: &Path,
) -> Result<Option<String>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    fs::remove_file(path).map_err(|error| error.to_string())?;
    let diagnostic = parse(&source)?;
    Ok(Some(render(&diagnostic, mode, binary)))
}

pub(super) fn signal_fallback(status: ExitStatus, binary: &Path, mode: DiagnosticsMode) -> String {
    let mut output = format!(
        "error[E0990]: native program terminated without a Severian runtime diagnostic\n note: process status was {status}\n help: rerun with `--diagnostics=internal`; if this repeats, report it as a compiler/runtime bug"
    );
    if mode.is_internal() {
        output.push_str(&format!(
            "\n internal: artifact={}, status={status:?}",
            binary.display()
        ));
    }
    output
}

fn parse(source: &str) -> Result<RuntimeDiagnostic, String> {
    let mut lines = source.lines();
    if lines.next() != Some(HEADER) {
        return Err("invalid Severian runtime diagnostic header".into());
    }
    let code = required(&mut lines, "code")?.to_owned();
    let message = required(&mut lines, "message")?.to_owned();
    let path = PathBuf::from(required(&mut lines, "path")?);
    let line = required(&mut lines, "line")?
        .parse::<usize>()
        .map_err(|_| "invalid Severian runtime diagnostic line".to_string())?;
    let column = required(&mut lines, "column")?
        .parse::<usize>()
        .map_err(|_| "invalid Severian runtime diagnostic column".to_string())?;
    let detail = lines.next().unwrap_or_default().to_owned();
    Ok(RuntimeDiagnostic {
        code,
        message,
        path,
        line,
        column,
        detail,
    })
}

fn required<'a>(lines: &mut impl Iterator<Item = &'a str>, field: &str) -> Result<&'a str, String> {
    lines
        .next()
        .ok_or_else(|| format!("Severian runtime diagnostic omitted {field}"))
}

fn render(diagnostic: &RuntimeDiagnostic, mode: DiagnosticsMode, binary: &Path) -> String {
    let display_path = logical_path(&diagnostic.path);
    let mut output = format!("error[{}]: {}", diagnostic.code, diagnostic.message);
    if !diagnostic.path.as_os_str().is_empty() && diagnostic.line > 0 {
        output.push_str(&format!(
            "\n --> {}:{}:{}",
            display_path.display(),
            diagnostic.line,
            diagnostic.column.max(1)
        ));
        if let Ok(source) = fs::read_to_string(&diagnostic.path) {
            if let Some(line) = source.lines().nth(diagnostic.line - 1) {
                output.push_str(&format!(
                    "\n   |\n{:>3} | {line}\n   | {}^",
                    diagnostic.line,
                    " ".repeat(diagnostic.column.saturating_sub(1))
                ));
            }
        }
    }
    if !diagnostic.detail.is_empty() {
        output.push_str(&format!("\n note: {}", diagnostic.detail));
    }
    output.push_str(&format!(
        "\n help: run `sev explain {}` for causes and fixes",
        diagnostic.code
    ));
    if mode.is_internal() {
        output.push_str(&format!(
            "\n internal: artifact={}, runtime-protocol=v1, source={}",
            binary.display(),
            diagnostic.path.display()
        ));
    }
    output
}

fn logical_path(path: &Path) -> PathBuf {
    if !path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .ok()
        .and_then(|directory| path.strip_prefix(directory).ok().map(Path::to_path_buf))
        .unwrap_or_else(|| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_renders_the_versioned_runtime_protocol() {
        let diagnostic = parse(
            "SEVERIAN_RUNTIME_DIAGNOSTIC_V1\nE0910\nindex is out of bounds\nsrc/main.sev\n4\n12\ncollection index 8 is invalid; length is 3\n",
        )
        .unwrap();
        let rendered = render(&diagnostic, DiagnosticsMode::User, Path::new("target/app"));
        assert!(rendered.starts_with("error[E0910]: index is out of bounds"));
        assert!(rendered.contains("src/main.sev:4:12"));
        assert!(rendered.contains("collection index 8 is invalid; length is 3"));
        assert!(rendered.contains("sev explain E0910"));
        assert!(!rendered.contains("internal:"));
    }
}
