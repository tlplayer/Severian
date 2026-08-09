use crate::{Diagnostic, DiagnosticBag, Severity};
use severian_source::SourceMap;

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub color: bool,
    pub context_lines: usize,
    pub show_codes: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            color: true,
            context_lines: 1,
            show_codes: true,
        }
    }
}

pub fn render_bag(
    bag: &DiagnosticBag,
    sources: Option<&SourceMap>,
    options: &RenderOptions,
) -> String {
    bag.diagnostics()
        .iter()
        .map(|diagnostic| render(diagnostic, sources, options))
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn render(
    diagnostic: &Diagnostic,
    sources: Option<&SourceMap>,
    options: &RenderOptions,
) -> String {
    let severity = match diagnostic.severity {
        Severity::Allow => "allow",
        Severity::Note => "note",
        Severity::Warning => "warning",
        Severity::Error => "error",
    };

    let prefix = if options.show_codes {
        format!("{severity}[{}]", diagnostic.code.0)
    } else {
        severity.to_owned()
    };

    let mut output = format!("{prefix}: {}", diagnostic.message);

    if let Some(source) = &diagnostic.source {
        output.push_str(&format!(
            "\n --> {}{}",
            source.file.display(),
            source
                .start_line
                .zip(source.start_column)
                .map(|(line, column)| format!(":{line}:{column}"))
                .unwrap_or_default()
        ));
    }

    if let (Some(sources), Some(source)) = (sources, diagnostic.source.as_ref()) {
        if let Some(file) = sources.file_by_path(&source.file) {
            if let (Some(start), Some(end)) = (
                file.location(source.start_byte),
                file.location(source.end_byte),
            ) {
                output.push('\n');
                let first = start.line.saturating_sub(options.context_lines as u32).max(1);
                let last = (end.line + options.context_lines as u32)
                    .min(file.line_count() as u32);

                for line in first..=last {
                    if let Some(text) = file.line(line) {
                        output.push_str(&format!("{line:>4} | {text}\n"));
                        if line == start.line {
                            let marker_start = start.column.saturating_sub(1) as usize;
                            let marker_len = if start.line == end.line {
                                end.column.saturating_sub(start.column).max(1) as usize
                            } else {
                                1
                            };
                            output.push_str(&format!(
                                "     | {}{}\n",
                                " ".repeat(marker_start),
                                "^".repeat(marker_len)
                            ));
                        }
                    }
                }
            }
        }
    }

    if let Some(help) = &diagnostic.help {
        output.push_str(&format!(" help: {help}"));
    }

    for related in &diagnostic.related {
        output.push_str(&format!(
            "\n note[{}]: {}",
            related.code.0,
            related.message
        ));
    }

    output.trim_end().to_owned()
}
