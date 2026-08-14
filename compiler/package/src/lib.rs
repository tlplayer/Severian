#![forbid(unsafe_code)]

use severian_ast::{FunctionDecl, ImportKind, Item, Module, Parameter, Span};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

mod install;
mod lockfile;
mod manifest;
mod policy;
mod resolution;
mod resolver;
mod sandbox;
mod signature;
mod system;
mod transport;
mod trust;

pub use install::{perform_installation, plan_installation, verify_installation, InstallationPlan};
pub use lockfile::{LockedExternal, LockedPackage, Lockfile};
pub use manifest::{InstallRequirement, InstallationManifest, SystemRequirement};
pub use policy::{
    BuildGate, BuildPolicy, CoveragePolicy, FileLimitException, FileLimitPolicy, MemoryPolicy,
};
pub use resolution::{
    publish_package, resolve_dependencies, resolve_dependencies_transient, update_dependencies,
    Resolution, ResolvedDependency,
};
pub use resolver::{signature_payload, InstallPlanItem, VendorCatalog, VendorPackage};
pub use sandbox::BuildSandbox;
pub use trust::{Date, TrustRegistry, TrustedPublisher};

pub const MANIFEST_FILE: &str = "package.toml";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionRule {
    pub function: String,
    pub runtime_symbol: String,
    pub opcode: u8,
    pub packing_bits: u8,
    pub max_chain: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionAlias {
    pub function: String,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphOperation {
    Input,
    Relu,
    Add,
    Matmul,
    Transpose,
    Scale,
    SoftmaxRows,
    LayerNorm,
    Run,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRule {
    pub function: String,
    pub operation: GraphOperation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompilerMetadata {
    pub symbols: HashMap<String, String>,
    pub external_functions: Vec<String>,
    pub fusion_rules: Vec<FusionRule>,
    pub fusion_aliases: Vec<FusionAlias>,
    pub graph_rules: Vec<GraphRule>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackageInterface {
    pub name: String,
    /// Package namespace that publicly exposes this package-local module's
    /// classes and traits. Local implementation functions retain `name`.
    pub export_package: Option<String>,
    pub module: Module,
    pub compiler: CompilerMetadata,
    pub source_path: PathBuf,
    pub source: String,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedOfficialPackage<'a> {
    pub name: &'a str,
    pub manifest: &'a str,
    pub source: &'a str,
    pub modules: &'a [EmbeddedOfficialModule<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedOfficialModule<'a> {
    pub path: &'a str,
    pub source: &'a str,
}

#[derive(Debug)]
pub enum PackageError {
    Io(std::io::Error),
    Manifest(String),
    Frontend {
        package: String,
        stage: &'static str,
        span: Span,
        message: String,
        source_path: Option<PathBuf>,
        source: Option<String>,
    },
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Manifest(message) => formatter.write_str(message),
            Self::Frontend {
                package,
                stage,
                span,
                message,
                source_path,
                source,
            } => {
                let prefix = if matches!(*stage, "lexer" | "parser") && !message.starts_with('E') {
                    "E000103: "
                } else {
                    ""
                };
                write!(
                    formatter,
                    "{prefix}invalid {stage} in package `{package}`: {message}"
                )?;
                if let (Some(path), Some(source)) = (source_path, source) {
                    let (line, column, text, marker) = source_location(source, *span);
                    let help = match *stage {
                        "lexer" | "parser" => {
                            "fix this dependency source or use a package version compatible with this compiler"
                        }
                        "unsafe policy" => {
                            "move host access behind a permitted library API; applications and tests remain safe"
                        }
                        "type safety" => {
                            "add a concrete type, or write `Any` explicitly at an intentional dynamic boundary"
                        }
                        _ => "run `sev explain <code>` for the normal repair",
                    };
                    write!(
                        formatter,
                        "\n --> {}:{line}:{column}\n{line:>4} | {text}\n     | {marker}\n help: {help}",
                        path.display()
                    )?;
                } else {
                    write!(
                        formatter,
                        " at bytes {}..{}\n help: rebuild or update the package with a compatible Severian compiler",
                        span.start, span.end
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PackageError {}

impl From<std::io::Error> for PackageError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

fn source_location(source: &str, span: Span) -> (usize, usize, &str, String) {
    let start = span.start.min(source.len());
    let end = span.end.min(source.len()).max(start);
    let prefix = source.get(..start).unwrap_or("");
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let line_end = source[start..]
        .find('\n')
        .map_or(source.len(), |length| start + length);
    let text = source.get(line_start..line_end).unwrap_or("");
    let column = source.get(line_start..start).unwrap_or("").chars().count() + 1;
    let marker_width = source
        .get(start..end.min(line_end))
        .unwrap_or("")
        .chars()
        .count()
        .max(1);
    let marker = format!("{}{}", " ".repeat(column - 1), "^".repeat(marker_width));
    (line, column, text, marker)
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn package_frontend_errors_render_source_context() {
        let error = PackageError::Frontend {
            package: "tensor".into(),
            stage: "parser",
            span: Span::new(17, 23),
            message: "expected a declaration or import".into(),
            source_path: Some(PathBuf::from("library/tensor/src/lib.sev")),
            source: Some("def valid():\n    broken\n".into()),
        };
        let rendered = error.to_string();
        assert!(rendered.contains("E000103"));
        assert!(rendered.contains("library/tensor/src/lib.sev:2:5"));
        assert!(rendered.contains("2 |     broken"));
        assert!(!rendered.contains("bytes 17..23"));
    }
}

mod interface;
mod safety;
mod target;

pub use interface::*;
pub use safety::*;
use safety::{enforce_manifest_unsafe_policy, with_frontend_source};
pub use target::*;

fn library_artifact_path(directory: &Path, package: &str) -> PathBuf {
    directory
        .join("target")
        .join("debug")
        .join("deps")
        .join(format!("lib{package}.sevi"))
}

fn package_name<'a>(manifest: &'a toml::Value, path: &Path) -> Result<&'a str, PackageError> {
    manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            PackageError::Manifest(format!("{} is missing `package.name`", path.display()))
        })
}

fn library_path(manifest: &toml::Value) -> &str {
    manifest
        .get("lib")
        .and_then(toml::Value::as_table)
        .and_then(|library| library.get("path"))
        .and_then(toml::Value::as_str)
        .unwrap_or("src/lib.sev")
}

fn parse_manifest(path: &Path) -> Result<toml::Value, PackageError> {
    let source = std::fs::read_to_string(path)?;
    let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
        PackageError::Manifest(format!("invalid manifest {}: {error}", path.display()))
    })?;
    manifest::validate_non_executable_manifest(&value, path)?;
    Ok(value)
}

fn manifest_in(directory: &Path) -> Option<PathBuf> {
    let manifest = directory.join(MANIFEST_FILE);
    manifest.is_file().then_some(manifest)
}
