use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiagnosticsMode {
    User,
    Internal,
}

impl DiagnosticsMode {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "user" => Ok(Self::User),
            "internal" => Ok(Self::Internal),
            _ => Err(format!(
                "unknown diagnostics mode `{value}`; use user or internal"
            )),
        }
    }

    pub(super) const fn is_internal(self) -> bool {
        matches!(self, Self::Internal)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BuildOptions {
    pub emit: String,
    pub target: String,
    pub max_errors: usize,
    pub message_format: String,
    pub verify_each: bool,
    pub diagnostics: DiagnosticsMode,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            emit: "executable".into(),
            target: "native".into(),
            max_errors: 50,
            message_format: "text".into(),
            verify_each: false,
            diagnostics: DiagnosticsMode::User,
        }
    }
}

pub(super) fn load(input: &Path) -> Result<BuildOptions, String> {
    let start = if input.is_file() {
        input.parent().unwrap_or(Path::new("."))
    } else {
        input
    };
    let Some(manifest) = severian_package::nearest_manifest(start) else {
        return Ok(BuildOptions::default());
    };
    let source = fs::read_to_string(&manifest).map_err(|error| error.to_string())?;
    let value = toml::from_str::<toml::Value>(&source)
        .map_err(|error| format!("invalid manifest {}: {error}", manifest.display()))?;
    let Some(build) = value.get("build").and_then(toml::Value::as_table) else {
        return Ok(BuildOptions::default());
    };

    let mut options = BuildOptions::default();
    options.emit = string(build, "emit", &options.emit, &manifest)?;
    options.target = string(build, "target", &options.target, &manifest)?;
    options.message_format = string(build, "message_format", &options.message_format, &manifest)?;
    options.max_errors = positive_integer(build, "max_errors", options.max_errors, &manifest)?;
    options.verify_each = boolean(build, "verify_each", options.verify_each, &manifest)?;
    let diagnostics = string(build, "diagnostics", "user", &manifest)?;
    options.diagnostics = DiagnosticsMode::parse(&diagnostics)
        .map_err(|error| format!("{}: {error}", manifest.display()))?;
    Ok(options)
}

fn string(
    table: &toml::Table,
    key: &str,
    default: &str,
    manifest: &Path,
) -> Result<String, String> {
    match table.get(key) {
        None => Ok(default.to_owned()),
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("{}.build.{key} must be a string", manifest.display())),
    }
}

fn positive_integer(
    table: &toml::Table,
    key: &str,
    default: usize,
    manifest: &Path,
) -> Result<usize, String> {
    match table.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_integer()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                format!(
                    "{}.build.{key} must be a positive integer",
                    manifest.display()
                )
            }),
    }
}

fn boolean(table: &toml::Table, key: &str, default: bool, manifest: &Path) -> Result<bool, String> {
    match table.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{}.build.{key} must be a boolean", manifest.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn manifest_values_supply_every_cli_build_default() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "severian-build-options-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("package.toml"),
            concat!(
                "[package]\nname = \"options\"\n",
                "[build]\n",
                "emit = \"mir\"\n",
                "target = \"xla\"\n",
                "max_errors = 7\n",
                "message_format = \"json\"\n",
                "verify_each = true\n",
                "diagnostics = \"internal\"\n",
            ),
        )
        .unwrap();

        let options = load(&root).unwrap();
        assert_eq!(options.emit, "mir");
        assert_eq!(options.target, "xla");
        assert_eq!(options.max_errors, 7);
        assert_eq!(options.message_format, "json");
        assert!(options.verify_each);
        assert_eq!(options.diagnostics, DiagnosticsMode::Internal);
        fs::remove_dir_all(root).unwrap();
    }
}
