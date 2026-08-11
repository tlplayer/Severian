use severian_source::{SourceMap, SourceSpan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirLocation {
    Unknown,
    FileLineColumn {
        file: String,
        line: u32,
        column: u32,
    },
    FileRange {
        file: String,
        start_line: u32,
        start_column: u32,
        end_line: u32,
        end_column: u32,
    },
    Named {
        name: String,
        child: Box<MlirLocation>,
    },
    Fused(Vec<MlirLocation>),
}

impl MlirLocation {
    pub fn from_source(sources: &SourceMap, span: SourceSpan) -> Self {
        let Some(file) = sources.file(span.file) else {
            return Self::Unknown;
        };
        let Some(start) = file.location(span.bytes.start) else {
            return Self::Unknown;
        };
        let Some(end) = file.location(span.bytes.end) else {
            return Self::Unknown;
        };

        if start.line == end.line && start.column == end.column {
            Self::FileLineColumn {
                file: file.path().to_string_lossy().into_owned(),
                line: start.line,
                column: start.column,
            }
        } else {
            Self::FileRange {
                file: file.path().to_string_lossy().into_owned(),
                start_line: start.line,
                start_column: start.column,
                end_line: end.line,
                end_column: end.column,
            }
        }
    }

    pub fn named(name: impl Into<String>, child: MlirLocation) -> Self {
        Self::Named {
            name: name.into(),
            child: Box::new(child),
        }
    }

    pub fn fused(locations: impl IntoIterator<Item = MlirLocation>) -> Self {
        Self::Fused(
            locations
                .into_iter()
                .filter(|location| !matches!(location, Self::Unknown))
                .collect(),
        )
    }

    pub fn render(&self) -> String {
        match self {
            Self::Unknown => "loc(?)".into(),
            Self::FileLineColumn { file, line, column } => {
                format!("loc(\"{}\":{}:{})", escape(file), line, column)
            }
            Self::FileRange {
                file,
                start_line,
                start_column,
                end_line,
                end_column,
            } => format!(
                "loc(\"{}\":{}:{} to {}:{})",
                escape(file),
                start_line,
                start_column,
                end_line,
                end_column
            ),
            Self::Named { name, child } => {
                format!("loc(\"{}\"({}))", escape(name), inner(child))
            }
            Self::Fused(locations) if locations.is_empty() => "loc(?)".into(),
            Self::Fused(locations) => {
                let values = locations.iter().map(inner).collect::<Vec<_>>().join(", ");
                format!("loc(fused[{values}])")
            }
        }
    }
}

pub fn attach_location(operation: &str, location: &MlirLocation) -> String {
    if operation.trim().is_empty() {
        return operation.to_owned();
    }
    format!("{} {}", operation.trim_end(), location.render())
}

fn inner(location: &MlirLocation) -> String {
    location
        .render()
        .strip_prefix("loc(")
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or("?")
        .to_owned()
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
