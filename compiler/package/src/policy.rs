use crate::{nearest_manifest, PackageError, MANIFEST_FILE};
use std::path::{Path, PathBuf};

mod architecture;

pub use architecture::{ArchitecturePolicy, ArchitectureRule, LayerPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildGate {
    Format,
    Lint,
    Compile,
    Architecture,
    Test,
    Profile,
    Coverage,
    Memory,
    Integration,
}

impl BuildGate {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Format => "format",
            Self::Lint => "lint",
            Self::Compile => "compile",
            Self::Architecture => "architecture",
            Self::Test => "test",
            Self::Profile => "profile",
            Self::Coverage => "coverage",
            Self::Memory => "memory",
            Self::Integration => "integ",
        }
    }

    fn parse(value: &str, manifest: &Path) -> Result<Self, PackageError> {
        match value {
            "format" | "fmt" => Ok(Self::Format),
            "lint" => Ok(Self::Lint),
            "compile" => Ok(Self::Compile),
            "architecture" => Ok(Self::Architecture),
            "test" | "unit" => Ok(Self::Test),
            "profile" | "speed" => Ok(Self::Profile),
            "coverage" => Ok(Self::Coverage),
            "memory" => Ok(Self::Memory),
            "integ" => Ok(Self::Integration),
            _ => Err(policy_error(
                manifest,
                format!("unknown build pipeline gate `{value}`"),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoveragePolicy {
    pub minimum: f64,
    pub changed_minimum: Option<f64>,
    pub regions: Option<f64>,
    pub branches: Option<f64>,
    pub functions: Option<f64>,
    pub per_file: bool,
    pub exclude: Vec<String>,
}

impl Default for CoveragePolicy {
    fn default() -> Self {
        Self {
            minimum: 75.0,
            changed_minimum: None,
            regions: None,
            branches: None,
            functions: None,
            per_file: false,
            exclude: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPolicy {
    pub leaks: bool,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self { leaks: true }
    }
}

/// Package-level boundary for information that must not escape semantic type
/// resolution. Packages can enable the switches incrementally while the
/// compiler keeps explicit source-level dynamic types legal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypeResolutionPolicy {
    pub deny_any: bool,
    pub deny_tensor_any: bool,
    pub deny_unresolved: bool,
    pub deny_inferred_fallback: bool,
    pub deny_lost_type_information: bool,
}

impl TypeResolutionPolicy {
    pub const fn is_permissive(self) -> bool {
        !self.deny_any
            && !self.deny_tensor_any
            && !self.deny_unresolved
            && !self.deny_inferred_fallback
            && !self.deny_lost_type_information
    }

    pub fn for_manifest(manifest: Option<&Path>) -> Result<Self, PackageError> {
        let Some(manifest) = manifest else {
            return Ok(Self::default());
        };
        let source = std::fs::read_to_string(manifest)?;
        let value = toml::from_str::<toml::Value>(&source)
            .map_err(|error| policy_error(manifest, format!("invalid {MANIFEST_FILE}: {error}")))?;
        parse_type_resolution(&value, manifest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLimitException {
    pub path: String,
    pub soft_lines: Option<usize>,
    pub hard_lines: usize,
    pub reason: String,
    pub expires: Option<String>,
    pub owner: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLimitPolicy {
    pub soft_lines: usize,
    pub hard_lines: usize,
    pub include: Vec<String>,
    pub exceptions: Vec<FileLimitException>,
}

impl Default for FileLimitPolicy {
    fn default() -> Self {
        Self {
            soft_lines: 500,
            hard_lines: 800,
            include: vec!["src/**/*.sev".into(), "tests/**/*.sev".into()],
            exceptions: Vec::new(),
        }
    }
}

impl FileLimitPolicy {
    pub fn limits_for(&self, relative_path: &str) -> (usize, usize, Option<&FileLimitException>) {
        let exception = self
            .exceptions
            .iter()
            .find(|exception| architecture_path_matches(&exception.path, relative_path));
        exception.map_or((self.soft_lines, self.hard_lines, None), |exception| {
            (
                exception.soft_lines.unwrap_or(self.soft_lines),
                exception.hard_lines,
                Some(exception),
            )
        })
    }

    pub fn includes(&self, relative_path: &str) -> bool {
        self.include
            .iter()
            .any(|pattern| architecture_path_matches(pattern, relative_path))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuildPolicy {
    pub root: PathBuf,
    pub manifest: Option<PathBuf>,
    pub pipeline: Vec<BuildGate>,
    pub coverage: CoveragePolicy,
    pub memory: MemoryPolicy,
    pub architecture: ArchitecturePolicy,
    pub files: FileLimitPolicy,
}

impl BuildPolicy {
    pub fn for_input(input: &Path) -> Result<Self, PackageError> {
        let start = if input.is_file() {
            input.parent().unwrap_or(Path::new("."))
        } else {
            input
        };
        let manifest = nearest_manifest(start);
        let root = manifest
            .as_ref()
            .and_then(|path| path.parent())
            .unwrap_or(start)
            .to_path_buf();
        let mut policy = Self {
            root,
            manifest: manifest.clone(),
            pipeline: default_pipeline(),
            coverage: CoveragePolicy::default(),
            memory: MemoryPolicy::default(),
            architecture: ArchitecturePolicy::default(),
            files: FileLimitPolicy::default(),
        };
        let Some(manifest) = manifest else {
            return Ok(policy);
        };
        let source = std::fs::read_to_string(&manifest)?;
        let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
            policy_error(&manifest, format!("invalid {MANIFEST_FILE}: {error}"))
        })?;
        policy.pipeline = parse_pipeline(&value, &manifest)?;
        policy.coverage = parse_coverage(&value, &manifest)?;
        policy.memory = parse_memory(&value, &manifest)?;
        policy.architecture = architecture::parse(&value, &manifest)?;
        policy.files = parse_file_limits(&value, &manifest)?;
        Ok(policy)
    }
}

fn default_pipeline() -> Vec<BuildGate> {
    vec![
        BuildGate::Format,
        BuildGate::Lint,
        BuildGate::Compile,
        BuildGate::Architecture,
        BuildGate::Test,
        BuildGate::Profile,
        BuildGate::Coverage,
        BuildGate::Memory,
        BuildGate::Integration,
    ]
}

fn parse_pipeline(value: &toml::Value, manifest: &Path) -> Result<Vec<BuildGate>, PackageError> {
    let Some(configured) = value
        .get("build")
        .and_then(toml::Value::as_table)
        .and_then(|build| build.get("pipeline"))
    else {
        return Ok(default_pipeline());
    };
    let entries = configured
        .as_array()
        .ok_or_else(|| policy_error(manifest, "`build.pipeline` must be an array of gate names"))?;
    let mut pipeline = Vec::new();
    for entry in entries {
        let name = entry.as_str().ok_or_else(|| {
            policy_error(manifest, "every `build.pipeline` entry must be a string")
        })?;
        let gate = BuildGate::parse(name, manifest)?;
        if pipeline.contains(&gate) {
            return Err(policy_error(
                manifest,
                format!(
                    "build pipeline gate `{}` appears more than once",
                    gate.name()
                ),
            ));
        }
        pipeline.push(gate);
    }
    let Some(compile_index) = pipeline.iter().position(|gate| *gate == BuildGate::Compile) else {
        return Err(policy_error(
            manifest,
            "`build.pipeline` must include `compile`",
        ));
    };
    if pipeline[..compile_index]
        .iter()
        .any(|gate| !matches!(gate, BuildGate::Format | BuildGate::Lint))
    {
        return Err(policy_error(
            manifest,
            "only `format` and `lint` may run before `compile`",
        ));
    }
    let canonical = default_pipeline();
    for required in canonical.iter().skip(2) {
        if !pipeline.contains(required) {
            return Err(policy_error(
                manifest,
                format!(
                    "build pipeline cannot omit mandatory `{}` gate",
                    required.name()
                ),
            ));
        }
    }
    let mut previous = 0;
    for gate in &pipeline {
        let position = canonical
            .iter()
            .position(|candidate| candidate == gate)
            .unwrap();
        if position < previous {
            return Err(policy_error(
                manifest,
                "build gates must follow format -> lint -> compile -> architecture -> test -> profile -> coverage -> memory -> integration",
            ));
        }
        previous = position;
    }
    Ok(pipeline)
}

fn parse_coverage(value: &toml::Value, manifest: &Path) -> Result<CoveragePolicy, PackageError> {
    let Some(table) = value.get("coverage").and_then(toml::Value::as_table) else {
        return Ok(CoveragePolicy::default());
    };
    for key in table.keys() {
        if !matches!(
            key.as_str(),
            "minimum"
                | "changed_minimum"
                | "changed-code-minimum"
                | "regions"
                | "branches"
                | "functions"
                | "per_file"
                | "exclude"
        ) {
            return Err(policy_error(
                manifest,
                format!("unknown `coverage` setting `{key}`"),
            ));
        }
    }
    let per_file = match table.get("per_file") {
        None => false,
        Some(toml::Value::Boolean(value)) => *value,
        Some(_) => {
            return Err(policy_error(
                manifest,
                "`coverage.per_file` must be a boolean",
            ))
        }
    };
    Ok(CoveragePolicy {
        minimum: percentage(table.get("minimum"), 75.0, manifest, "coverage.minimum")?,
        changed_minimum: optional_percentage(
            table
                .get("changed_minimum")
                .or_else(|| table.get("changed-code-minimum")),
            manifest,
            "coverage.changed_minimum",
        )?,
        regions: optional_percentage(table.get("regions"), manifest, "coverage.regions")?,
        branches: optional_percentage(table.get("branches"), manifest, "coverage.branches")?,
        functions: optional_percentage(table.get("functions"), manifest, "coverage.functions")?,
        per_file,
        exclude: string_array(table.get("exclude"), manifest, "coverage.exclude")?
            .unwrap_or_default(),
    })
}

fn parse_memory(value: &toml::Value, manifest: &Path) -> Result<MemoryPolicy, PackageError> {
    let Some(table) = value.get("memory").and_then(toml::Value::as_table) else {
        return Ok(MemoryPolicy::default());
    };
    let leaks = match table.get("leaks") {
        None => true,
        Some(toml::Value::Boolean(value)) => *value,
        Some(toml::Value::String(value)) if value == "deny" => true,
        Some(toml::Value::String(value)) if value == "allow" => false,
        _ => {
            return Err(policy_error(
                manifest,
                "`memory.leaks` must be `deny`, `allow`, true, or false",
            ))
        }
    };
    Ok(MemoryPolicy { leaks })
}

fn parse_type_resolution(
    value: &toml::Value,
    manifest: &Path,
) -> Result<TypeResolutionPolicy, PackageError> {
    let Some(compiler) = value.get("compiler") else {
        return Ok(TypeResolutionPolicy::default());
    };
    let compiler = compiler
        .as_table()
        .ok_or_else(|| policy_error(manifest, "`compiler` must be a table"))?;
    let Some(configured) = compiler.get("type_resolution") else {
        return Ok(TypeResolutionPolicy::default());
    };
    let table = configured
        .as_table()
        .ok_or_else(|| policy_error(manifest, "`compiler.type_resolution` must be a table"))?;
    for key in table.keys() {
        if !matches!(
            key.as_str(),
            "deny_any"
                | "deny_tensor_any"
                | "deny_unresolved"
                | "deny_inferred_fallback"
                | "deny_lost_type_information"
        ) {
            return Err(policy_error(
                manifest,
                format!("unknown `compiler.type_resolution` setting `{key}`"),
            ));
        }
    }
    let flag = |name: &str| -> Result<bool, PackageError> {
        match table.get(name) {
            None => Ok(false),
            Some(toml::Value::Boolean(value)) => Ok(*value),
            Some(_) => Err(policy_error(
                manifest,
                format!("`compiler.type_resolution.{name}` must be a boolean"),
            )),
        }
    };
    Ok(TypeResolutionPolicy {
        deny_any: flag("deny_any")?,
        deny_tensor_any: flag("deny_tensor_any")?,
        deny_unresolved: flag("deny_unresolved")?,
        deny_inferred_fallback: flag("deny_inferred_fallback")?,
        deny_lost_type_information: flag("deny_lost_type_information")?,
    })
}

fn parse_file_limits(
    value: &toml::Value,
    manifest: &Path,
) -> Result<FileLimitPolicy, PackageError> {
    let Some(table) = value
        .get("architecture")
        .and_then(toml::Value::as_table)
        .and_then(|architecture| architecture.get("files"))
        .and_then(toml::Value::as_table)
    else {
        return Ok(FileLimitPolicy::default());
    };
    let soft_lines = positive_integer(
        table.get("soft_lines"),
        500,
        manifest,
        "architecture.files.soft_lines",
    )?;
    let hard_lines = positive_integer(
        table.get("hard_lines"),
        800,
        manifest,
        "architecture.files.hard_lines",
    )?;
    if soft_lines > hard_lines {
        return Err(policy_error(
            manifest,
            "file soft line limit cannot exceed the hard limit",
        ));
    }
    let include = string_array(table.get("include"), manifest, "architecture.files.include")?
        .unwrap_or_else(|| FileLimitPolicy::default().include);
    let exceptions = value
        .get("architecture")
        .and_then(toml::Value::as_table)
        .and_then(|architecture| architecture.get("files"))
        .and_then(toml::Value::as_table)
        .and_then(|files| files.get("exception"))
        .and_then(toml::Value::as_array)
        .map_or(Ok(Vec::new()), |entries| {
            entries
                .iter()
                .map(|entry| parse_exception(entry, manifest, soft_lines))
                .collect()
        })?;
    Ok(FileLimitPolicy {
        soft_lines,
        hard_lines,
        include,
        exceptions,
    })
}

fn parse_exception(
    value: &toml::Value,
    manifest: &Path,
    default_soft: usize,
) -> Result<FileLimitException, PackageError> {
    let table = value.as_table().ok_or_else(|| {
        policy_error(
            manifest,
            "every architecture file exception must be a table",
        )
    })?;
    let required = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
            .ok_or_else(|| policy_error(manifest, format!("file exception `{key}` is required")))
    };
    let path = required("path")?;
    let reason = required("reason")?;
    let hard_lines = positive_integer(
        table.get("hard_lines"),
        0,
        manifest,
        "file exception hard_lines",
    )?;
    let soft_lines = table
        .get("soft_lines")
        .map(|value| {
            positive_integer(
                Some(value),
                default_soft,
                manifest,
                "file exception soft_lines",
            )
        })
        .transpose()?;
    if soft_lines.is_some_and(|soft| soft > hard_lines) {
        return Err(policy_error(
            manifest,
            format!("file exception `{path}` has a soft limit above its hard limit"),
        ));
    }
    Ok(FileLimitException {
        path,
        soft_lines,
        hard_lines,
        reason,
        expires: table
            .get("expires")
            .and_then(toml::Value::as_str)
            .map(str::to_owned),
        owner: table
            .get("owner")
            .and_then(toml::Value::as_str)
            .map(str::to_owned),
    })
}

fn percentage(
    value: Option<&toml::Value>,
    default: f64,
    manifest: &Path,
    name: &str,
) -> Result<f64, PackageError> {
    optional_percentage(value, manifest, name).map(|value| value.unwrap_or(default))
}

fn optional_percentage(
    value: Option<&toml::Value>,
    manifest: &Path,
    name: &str,
) -> Result<Option<f64>, PackageError> {
    let Some(value) = value else { return Ok(None) };
    let number = value
        .as_float()
        .or_else(|| value.as_integer().map(|value| value as f64));
    match number {
        Some(value) if (0.0..=100.0).contains(&value) => Ok(Some(value)),
        _ => Err(policy_error(
            manifest,
            format!("`{name}` must be between 0 and 100"),
        )),
    }
}

fn positive_integer(
    value: Option<&toml::Value>,
    default: usize,
    manifest: &Path,
    name: &str,
) -> Result<usize, PackageError> {
    let value = value
        .and_then(toml::Value::as_integer)
        .unwrap_or(default as i64);
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| policy_error(manifest, format!("`{name}` must be a positive integer")))
}

fn string_array(
    value: Option<&toml::Value>,
    manifest: &Path,
    name: &str,
) -> Result<Option<Vec<String>>, PackageError> {
    let Some(value) = value else { return Ok(None) };
    value
        .as_array()
        .ok_or_else(|| policy_error(manifest, format!("`{name}` must be an array of strings")))?
        .iter()
        .map(|entry| {
            entry.as_str().map(str::to_owned).ok_or_else(|| {
                policy_error(manifest, format!("every `{name}` entry must be a string"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub fn architecture_path_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.trim_start_matches("./");
    let path = path.trim_start_matches("./");
    let pattern = pattern.split('/').collect::<Vec<_>>();
    let path = path.split('/').collect::<Vec<_>>();
    match_path_components(&pattern, &path)
}

fn match_path_components(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => {
            match_path_components(rest, path)
                || (!path.is_empty() && match_path_components(pattern, &path[1..]))
        }
        Some((component, rest)) => {
            !path.is_empty()
                && match_component(component.as_bytes(), path[0].as_bytes())
                && match_path_components(rest, &path[1..])
        }
    }
}

fn match_component(pattern: &[u8], value: &[u8]) -> bool {
    match pattern.split_first() {
        None => value.is_empty(),
        Some((&b'*', rest)) => {
            match_component(rest, value)
                || (!value.is_empty() && match_component(pattern, &value[1..]))
        }
        Some((&expected, rest)) => value
            .split_first()
            .is_some_and(|(&actual, tail)| expected == actual && match_component(rest, tail)),
    }
}

fn policy_error(manifest: &Path, message: impl std::fmt::Display) -> PackageError {
    PackageError::Manifest(format!("{}: {message}", manifest.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn recursive_patterns_cover_source_files() {
        assert!(architecture_path_matches(
            "compiler/**/*.rs",
            "compiler/hir/src/lib.rs"
        ));
        assert!(architecture_path_matches(
            "src/**/*.sev",
            "src/model/main.sev"
        ));
        assert!(!architecture_path_matches("src/**/*.sev", "tests/main.sev"));
    }

    #[test]
    fn package_cannot_remove_mandatory_build_gates() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "severian-build-policy-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(MANIFEST_FILE),
            "[package]\nname = \"gate-test\"\n\n[build]\npipeline = [\"compile\", \"test\"]\n",
        )
        .unwrap();

        let error = BuildPolicy::for_input(&root).unwrap_err().to_string();
        assert!(error.contains("cannot omit mandatory `architecture` gate"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn default_pipeline_applies_format_and_lint_before_compilation() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "severian-default-build-policy-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(MANIFEST_FILE),
            "[package]\nname = \"gate-test\"\n",
        )
        .unwrap();

        let policy = BuildPolicy::for_input(&root).unwrap();
        assert_eq!(policy.pipeline[0], BuildGate::Format);
        assert_eq!(policy.pipeline[1], BuildGate::Lint);
        assert_eq!(policy.pipeline[2], BuildGate::Compile);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_pipeline_can_disable_automatic_source_mutation() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "severian-explicit-build-policy-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(MANIFEST_FILE),
            concat!(
                "[package]\nname = \"gate-test\"\n\n",
                "[build]\npipeline = [\n",
                "  \"compile\", \"architecture\", \"test\", \"profile\",\n",
                "  \"coverage\", \"memory\", \"integration\",\n",
                "]\n",
            ),
        )
        .unwrap();

        let policy = BuildPolicy::for_input(&root).unwrap();
        assert_eq!(policy.pipeline.first(), Some(&BuildGate::Compile));
        assert!(!policy.pipeline.contains(&BuildGate::Format));
        assert!(!policy.pipeline.contains(&BuildGate::Lint));
        let _ = std::fs::remove_dir_all(root);
    }
}
