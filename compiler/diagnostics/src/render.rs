use crate::{
    Applicability, Diagnostic, DiagnosticBag, DiagnosticLabel, LabelStyle, Severity, SourceRange,
    TextEdit,
};
use severian_source::SourceMap;
use std::fmt::Write;
use std::path::Path;

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
        render_source(&mut output, source, None, '^', sources, options);
    }
    for label in &diagnostic.labels {
        render_label(&mut output, label, sources, options);
    }

    for note in &diagnostic.notes {
        output.push_str(&format!("\n note: {note}"));
    }
    if let Some(help) = &diagnostic.help {
        output.push_str(&format!("\n help: {help}"));
    }
    for suggestion in &diagnostic.suggestions {
        output.push_str(&format!("\n help: {}", suggestion.message));
        for edit in &suggestion.edits {
            render_edit(&mut output, edit, sources);
        }
    }

    for related in &diagnostic.related {
        output.push_str(&format!("\n note[{}]: {}", related.code.0, related.message));
    }

    output.trim_end().to_owned()
}

/// Serializes the full diagnostic protocol used by editors and build agents.
/// `rendered` preserves the human-facing message while the remaining fields
/// are stable, structured data suitable for quick fixes.
pub fn render_json(diagnostic: &Diagnostic, rendered: &str, fallback_path: &Path) -> String {
    let severity = match diagnostic.severity {
        Severity::Allow => "allow",
        Severity::Note => "note",
        Severity::Warning => "warning",
        Severity::Error => "error",
    };
    let primary_path = diagnostic
        .labels
        .iter()
        .find(|label| label.style == LabelStyle::Primary)
        .map(|label| label.source.file.as_path())
        .or_else(|| {
            diagnostic
                .source
                .as_ref()
                .map(|source| source.file.as_path())
        })
        .unwrap_or(fallback_path);
    let mut labels = Vec::new();
    if let Some(source) = &diagnostic.source {
        labels.push(label_json(&DiagnosticLabel {
            style: LabelStyle::Primary,
            source: source.clone(),
            message: None,
        }));
    }
    labels.extend(diagnostic.labels.iter().map(label_json));
    let labels = labels.join(",");
    let notes = diagnostic
        .notes
        .iter()
        .map(|note| format!("\"{}\"", json_escape(note)))
        .collect::<Vec<_>>()
        .join(",");
    let suggestions = diagnostic
        .suggestions
        .iter()
        .map(|suggestion| {
            let applicability = match suggestion.applicability {
                Applicability::MachineApplicable => "machine-applicable",
                Applicability::MaybeIncorrect => "maybe-incorrect",
            };
            let edits = suggestion
                .edits
                .iter()
                .map(edit_json)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"message\":\"{}\",\"applicability\":\"{applicability}\",\"edits\":[{edits}]}}",
                json_escape(&suggestion.message)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let related = diagnostic
        .related
        .iter()
        .map(|related| {
            format!(
                "{{\"code\":\"{}\",\"message\":\"{}\"}}",
                json_escape(&related.code.0),
                json_escape(&related.message),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"severity\":\"{severity}\",\"code\":\"{}\",\"path\":\"{}\",\"message\":\"{}\",\"rendered\":\"{}\",\"help\":{},\"labels\":[{labels}],\"notes\":[{notes}],\"suggestions\":[{suggestions}],\"related\":[{related}]}}",
        json_escape(&diagnostic.code.0),
        json_escape(&primary_path.display().to_string()),
        json_escape(&diagnostic.message),
        json_escape(rendered),
        optional_json_string(diagnostic.help.as_deref()),
    )
}

fn label_json(label: &DiagnosticLabel) -> String {
    let style = match label.style {
        LabelStyle::Primary => "primary",
        LabelStyle::Secondary => "secondary",
    };
    format!(
        "{{\"style\":\"{style}\",\"message\":{},\"range\":{}}}",
        optional_json_string(label.message.as_deref()),
        range_json(&label.source),
    )
}

fn edit_json(edit: &TextEdit) -> String {
    format!(
        "{{\"range\":{},\"replacement\":\"{}\"}}",
        range_json(&edit.source),
        json_escape(&edit.replacement),
    )
}

fn range_json(range: &SourceRange) -> String {
    format!(
        "{{\"path\":\"{}\",\"start\":{{\"byte\":{},\"line\":{},\"column\":{}}},\"end\":{{\"byte\":{},\"line\":{},\"column\":{}}}}}",
        json_escape(&range.file.display().to_string()),
        range.start_byte,
        optional_number(range.start_line),
        optional_number(range.start_column),
        range.end_byte,
        optional_number(range.end_line),
        optional_number(range.end_column),
    )
}

fn optional_number(value: Option<u32>) -> String {
    value.map_or_else(|| "null".into(), |value| value.to_string())
}

fn optional_json_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".into(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                write!(escaped, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn render_label(
    output: &mut String,
    label: &DiagnosticLabel,
    sources: Option<&SourceMap>,
    options: &RenderOptions,
) {
    let marker = match label.style {
        LabelStyle::Primary => '^',
        LabelStyle::Secondary => '-',
    };
    render_source(
        output,
        &label.source,
        label.message.as_deref(),
        marker,
        sources,
        options,
    );
}

fn render_source(
    output: &mut String,
    source: &SourceRange,
    label: Option<&str>,
    marker: char,
    sources: Option<&SourceMap>,
    options: &RenderOptions,
) {
    output.push_str(&format!(
        "\n --> {}{}",
        source.file.display(),
        source
            .start_line
            .zip(source.start_column)
            .map(|(line, column)| format!(":{line}:{column}"))
            .unwrap_or_default()
    ));
    let Some(file) = sources.and_then(|sources| sources.file_by_path(&source.file)) else {
        if let Some(label) = label {
            output.push_str(&format!("\n     = {label}"));
        }
        return;
    };
    let (Some(start), Some(end)) = (
        file.location(source.start_byte),
        file.location(source.end_byte),
    ) else {
        return;
    };
    output.push_str("\n     |");
    let first = start
        .line
        .saturating_sub(options.context_lines as u32)
        .max(1);
    let last = (end.line + options.context_lines as u32).min(file.line_count() as u32);
    for line in first..=last {
        let Some(text) = file.line(line) else {
            continue;
        };
        output.push_str(&format!("\n{line:>4} | {text}"));
        if line == start.line {
            let marker_start = start.column.saturating_sub(1) as usize;
            let marker_len = if start.line == end.line {
                end.column.saturating_sub(start.column).max(1) as usize
            } else {
                1
            };
            output.push_str(&format!(
                "\n     | {}{}{}",
                " ".repeat(marker_start),
                marker.to_string().repeat(marker_len),
                label.map_or(String::new(), |label| format!(" {label}"))
            ));
        }
    }
}

fn render_edit(output: &mut String, edit: &TextEdit, sources: Option<&SourceMap>) {
    let Some(file) = sources.and_then(|sources| sources.file_by_path(&edit.source.file)) else {
        return;
    };
    let Some(start) = file.location(edit.source.start_byte) else {
        return;
    };
    let Some(line) = file.line(start.line) else {
        return;
    };
    let Some(end) = file.location(edit.source.end_byte) else {
        return;
    };
    if end.line != start.line {
        return;
    }
    let byte_at_column = |column: u32| {
        line.char_indices()
            .nth(column.saturating_sub(1) as usize)
            .map_or(line.len(), |(byte, _)| byte)
    };
    let relative_start = byte_at_column(start.column);
    let relative_end = byte_at_column(end.column);
    let mut corrected = line.to_owned();
    corrected.replace_range(relative_start..relative_end, &edit.replacement);
    let marker_width = edit.replacement.chars().count().max(1);
    output.push_str(&format!(
        "\n     |\n{:>4} | {corrected}\n     | {}{}",
        start.line,
        " ".repeat(start.column.saturating_sub(1) as usize),
        "+".repeat(marker_width)
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Applicability, DiagnosticSuggestion};
    use std::path::PathBuf;

    fn range(path: &str, start: usize, end: usize) -> SourceRange {
        SourceRange {
            file: PathBuf::from(path),
            start_byte: start,
            end_byte: end,
            start_line: None,
            start_column: None,
            end_line: None,
            end_column: None,
        }
    }

    #[test]
    fn renders_primary_secondary_and_machine_applicable_edits() {
        let source = "def load(name: string, device: string):\n    return\n\ndef main():\n    load(\"Qwen\")\n";
        let path = "src/model.sev";
        let call_start = source.find("load(\"Qwen\")").unwrap();
        let insertion = call_start + "load(\"Qwen\"".len();
        let requirement_start = source.find("device: string").unwrap();
        let requirement = range(path, requirement_start, requirement_start + "device".len());
        let call = range(path, call_start, call_start + "load(\"Qwen\")".len());
        let edit = range(path, insertion, insertion);
        let mut sources = SourceMap::new();
        sources.add(path, source);
        let diagnostic = Diagnostic::error("E000203", "missing argument `device`")
            .with_label(DiagnosticLabel::primary(call, "argument is required here"))
            .with_label(DiagnosticLabel::secondary(
                requirement,
                "`device` is declared without a default",
            ))
            .with_note("the call supplies 1 of 2 required arguments")
            .with_suggestion(DiagnosticSuggestion {
                message: "add `device`".into(),
                applicability: Applicability::MachineApplicable,
                edits: vec![TextEdit {
                    source: edit,
                    replacement: ", device = \"cpu\"".into(),
                }],
            });
        let rendered = render(
            &diagnostic,
            Some(&sources),
            &RenderOptions {
                color: false,
                context_lines: 0,
                show_codes: true,
            },
        );
        assert!(rendered.starts_with("error[E000203]: missing argument `device`"));
        assert!(rendered.contains("^^^^ argument is required here"));
        assert!(rendered.contains("------ `device` is declared without a default"));
        assert!(rendered.contains("note: the call supplies 1 of 2 required arguments"));
        assert!(rendered.contains("help: add `device`"));
        assert!(rendered.contains("load(\"Qwen\", device = \"cpu\")"));

        let json = render_json(&diagnostic, &rendered, Path::new(path));
        assert!(json.contains("\"code\":\"E000203\""));
        assert!(json.contains("\"style\":\"secondary\""));
        assert!(json.contains("\"applicability\":\"machine-applicable\""));
        assert!(json.contains("\"replacement\":\", device = \\\"cpu\\\"\""));
    }
}
