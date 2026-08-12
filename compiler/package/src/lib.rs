#![forbid(unsafe_code)]

use severian_ast::{FunctionDecl, ImportKind, Item, Module, Parameter, Span};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

mod install;
mod lockfile;
mod manifest;
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
                    "E0103: "
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
        assert!(rendered.contains("E0103"));
        assert!(rendered.contains("library/tensor/src/lib.sev:2:5"));
        assert!(rendered.contains("2 |     broken"));
        assert!(!rendered.contains("bytes 17..23"));
    }
}

pub fn find_manifest(source: &Path) -> Option<PathBuf> {
    source.parent()?.ancestors().find_map(manifest_in)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryTarget {
    pub name: String,
    pub source: PathBuf,
    pub package_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryTarget {
    pub name: String,
    pub source: PathBuf,
    pub artifact: PathBuf,
    pub manifest: PathBuf,
}

pub fn nearest_manifest(directory: &Path) -> Option<PathBuf> {
    directory.ancestors().find_map(manifest_in)
}

pub fn workspace_manifests(directory: &Path) -> Result<Vec<PathBuf>, PackageError> {
    let manifest_path = nearest_manifest(directory).ok_or_else(|| {
        PackageError::Manifest(format!(
            "could not find package.toml from {}",
            directory.display()
        ))
    })?;
    let manifest = parse_manifest(&manifest_path)?;
    let Some(members) = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
    else {
        return Ok(vec![manifest_path]);
    };
    let root = manifest_path
        .parent()
        .expect("a manifest path has a parent");
    members
        .iter()
        .map(|member| {
            let member = member.as_str().ok_or_else(|| {
                PackageError::Manifest("workspace members must be paths encoded as strings".into())
            })?;
            let directory = root.join(member);
            let path = manifest_in(&directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
            if !path.is_file() {
                return Err(PackageError::Manifest(format!(
                    "workspace member manifest {} does not exist",
                    path.display()
                )));
            }
            Ok(path)
        })
        .collect()
}

pub fn library_build_plan(manifest_path: &Path) -> Result<Vec<LibraryTarget>, PackageError> {
    let resolution = resolve_dependencies(manifest_path)?;
    let mut targets = Vec::new();
    for dependency in resolution.dependencies {
        let manifest = parse_manifest(&dependency.manifest)?;
        let source = dependency.root.join(library_path(&manifest));
        if manifest.get("lib").is_some() || source.is_file() {
            targets.push(LibraryTarget {
                name: dependency.import_name,
                source,
                artifact: library_artifact_path(&dependency.root, &dependency.package_name),
                manifest: dependency.manifest,
            });
        }
    }
    Ok(targets)
}

pub fn write_library_artifact(target: &LibraryTarget) -> Result<(), PackageError> {
    let source = std::fs::read_to_string(&target.source)?;
    if let Some(parent) = target.artifact.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &target.artifact,
        format!(
            "# severian-library-artifact v1\n# package {}\n{}",
            target.name,
            strip_package_tests(&source)
        ),
    )?;
    Ok(())
}

fn strip_package_tests(source: &str) -> String {
    let mut output = String::new();
    let mut skipped_test_indent = None;
    for line in source.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if let Some(test_indent) = skipped_test_indent {
            if trimmed.is_empty() || indent > test_indent {
                continue;
            }
            skipped_test_indent = None;
        }
        if (trimmed == "test:" || trimmed.starts_with("test ")) && trimmed.ends_with(':') {
            skipped_test_indent = Some(indent);
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    output
}

/// Resolves the default binary target using Cargo-compatible `[package]` and
/// `[[bin]]` manifest fields.
pub fn default_binary_target(directory: &Path) -> Result<BinaryTarget, PackageError> {
    let direct = directory.join("main.sev");
    let manifest_path = nearest_manifest(directory);
    let Some(manifest_path) = manifest_path else {
        if direct.is_file() {
            return Ok(BinaryTarget {
                name: "main".into(),
                source: direct,
                package_root: directory.to_path_buf(),
            });
        }
        return Err(PackageError::Manifest(format!(
            "could not find `main.sev` or package.toml from {}",
            directory.display()
        )));
    };
    let manifest = parse_manifest(&manifest_path)?;
    let package_root = manifest_path
        .parent()
        .expect("a manifest path has a parent")
        .to_path_buf();
    if manifest.get("package").is_none() {
        let member = manifest
            .get("workspace")
            .and_then(toml::Value::as_table)
            .and_then(|workspace| workspace.get("members"))
            .and_then(toml::Value::as_array)
            .and_then(|members| members.first())
            .and_then(toml::Value::as_str)
            .ok_or_else(|| {
                PackageError::Manifest(format!(
                    "workspace manifest {} has no package or members",
                    manifest_path.display()
                ))
            })?;
        return default_binary_target(&package_root.join(member));
    }
    let binary = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .and_then(|binaries| binaries.first())
        .and_then(toml::Value::as_table);
    let binary_path = binary
        .and_then(|binary| binary.get("path"))
        .and_then(toml::Value::as_str)
        .unwrap_or("src/main.sev");
    let source = package_root.join(binary_path);
    if !source.is_file() {
        return Err(PackageError::Manifest(format!(
            "binary source {} does not exist",
            source.display()
        )));
    }
    let name = binary
        .and_then(|binary| binary.get("name"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            manifest
                .get("package")
                .and_then(toml::Value::as_table)
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "main".into());
    Ok(BinaryTarget {
        name,
        source,
        package_root,
    })
}

/// Resolves every binary built by `sev build` from the nearest package or
/// workspace manifest. Workspace member syntax intentionally matches Cargo's
/// string-array form.
pub fn workspace_binary_targets(directory: &Path) -> Result<Vec<BinaryTarget>, PackageError> {
    let direct = directory.join("main.sev");
    let manifest_path = nearest_manifest(directory);
    let Some(manifest_path) = manifest_path else {
        if direct.is_file() {
            return Ok(vec![default_binary_target(directory)?]);
        }
        return Err(PackageError::Manifest(format!(
            "could not find `main.sev` or package.toml from {}",
            directory.display()
        )));
    };
    let manifest = parse_manifest(&manifest_path)?;
    if manifest.get("package").is_some() {
        let root = manifest_path
            .parent()
            .expect("a manifest path has a parent");
        let has_binary = manifest.get("bin").is_some()
            || root.join("src/main.sev").is_file()
            || root.join("main.sev").is_file();
        if !has_binary {
            return Ok(Vec::new());
        }
        return Ok(vec![default_binary_target(directory)?]);
    }
    let root = manifest_path
        .parent()
        .expect("a manifest path has a parent");
    let members = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "workspace manifest {} has no members",
                manifest_path.display()
            ))
        })?;
    let mut targets = Vec::new();
    for member in members {
        let member = member.as_str().ok_or_else(|| {
            PackageError::Manifest("workspace members must be paths encoded as strings".into())
        })?;
        let member_root = root.join(member);
        let member_manifest_path =
            manifest_in(&member_root).unwrap_or_else(|| member_root.join(MANIFEST_FILE));
        let member_manifest = parse_manifest(&member_manifest_path)?;
        let has_binary = member_manifest.get("bin").is_some()
            || member_root.join("src/main.sev").is_file()
            || member_root.join("main.sev").is_file();
        if has_binary {
            let mut target = default_binary_target(&member_root)?;
            target.package_root = root.to_path_buf();
            targets.push(target);
        }
    }
    Ok(targets)
}

/// Resolves the first binary target for `sev build` from the nearest manifest.
pub fn default_binary_source(directory: &Path) -> Result<PathBuf, PackageError> {
    Ok(default_binary_target(directory)?.source)
}

pub fn load_path_dependency_sources(manifest_path: &Path) -> Result<Vec<String>, PackageError> {
    resolve_dependencies(manifest_path)?
        .dependencies
        .into_iter()
        .map(|dependency| {
            let manifest = parse_manifest(&dependency.manifest)?;
            let source = dependency.root.join(library_path(&manifest));
            std::fs::read_to_string(&source).map_err(|error| {
                PackageError::Manifest(format!(
                    "could not read library for `{}` at {}: {error}",
                    dependency.import_name,
                    source.display()
                ))
            })
        })
        .collect()
}

/// Enforces capability-scoped unsafe exceptions from `package.toml`. Unsafe is
/// denied by default. Native ABI declarations remain library-only; selected
/// examples may opt individual source files into language capabilities such as
/// `pointers` or `runtime-owned-tasks`. Tests are rejected by the parser even
/// when their package has an exception.
pub fn enforce_unsafe_policy(
    manifest_path: Option<&Path>,
    source_path: &Path,
    source: &str,
) -> Result<(), PackageError> {
    let tokens = severian_lexer::lex(source).map_err(|error| PackageError::Frontend {
        package: package_label(manifest_path),
        stage: "lexer",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.to_path_buf()),
        source: Some(source.to_owned()),
    })?;
    let Some(token) = tokens
        .iter()
        .find(|token| token.kind == severian_lexer::TokenKind::Unsafe)
    else {
        return Ok(());
    };
    let Some(manifest_path) = manifest_path else {
        return Err(with_frontend_source(
            unsafe_policy_error("source without package.toml", token.span, "source-file"),
            source_path,
            source,
        ));
    };
    let manifest = parse_manifest(manifest_path)?;
    enforce_manifest_unsafe_policy(
        &manifest,
        manifest_path,
        source_path,
        &tokens,
        token.span,
        false,
    )
    .map_err(|error| with_frontend_source(error, source_path, source))
}

/// Rejects declaration sites that would silently fall back to `Any` when the
/// package opts into `[package] type-safe = true`.
///
/// Explicit `Any` remains available as an intentional escape hatch. The check
/// is deliberately package-scoped so exploratory packages can stay dynamic
/// while stable libraries progressively add concrete boundary types.
pub fn enforce_type_safe_policy(
    manifest_path: Option<&Path>,
    source_path: &Path,
    module: &Module,
    source: &str,
) -> Result<(), PackageError> {
    let Some(manifest_path) = manifest_path else {
        return Ok(());
    };
    let manifest = parse_manifest(manifest_path)?;
    let Some(enabled) = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("type-safe"))
    else {
        return Ok(());
    };
    let enabled = enabled.as_bool().ok_or_else(|| {
        PackageError::Manifest(format!(
            "{}.package.type-safe must be a boolean",
            manifest_path.display()
        ))
    })?;
    if !enabled {
        return Ok(());
    }

    let package = package_name(&manifest, manifest_path)?.to_owned();
    if let Some((span, declaration, kind)) = first_inferred_any(module) {
        let snippet = source_snippet(source, span);
        return Err(PackageError::Frontend {
            package,
            stage: "type safety",
            span,
            message: format!(
                "E0201: {kind} `{declaration}` defaults to `Any`\n\
                 source: {snippet}\n\
                 add an explicit type, such as `{declaration}: ConcreteType`; write `{declaration}: Any` only when dynamic typing is intentional"
            ),
            source_path: Some(source_path.to_path_buf()),
            source: Some(source.to_owned()),
        });
    }
    Ok(())
}

fn first_inferred_any(module: &Module) -> Option<(Span, &str, &'static str)> {
    for item in &module.items {
        match item {
            Item::Function(function) => {
                if let Some(problem) = untyped_function_parameter(function) {
                    return Some(problem);
                }
            }
            Item::Class(class) => {
                if let Some(field) = class.fields.iter().find(|field| field.ty.is_none()) {
                    return Some((field.name.span, &field.name.name, "field"));
                }
                for constructor in &class.constructors {
                    if let Some(parameter) = untyped_parameter(&constructor.params) {
                        return Some((parameter.name.span, &parameter.name.name, "parameter"));
                    }
                }
                for method in &class.methods {
                    if let Some(problem) = untyped_function_parameter(method) {
                        return Some(problem);
                    }
                }
            }
            Item::Trait(trait_declaration) => {
                for method in &trait_declaration.methods {
                    if let Some(parameter) = untyped_parameter(&method.params) {
                        return Some((parameter.name.span, &parameter.name.name, "parameter"));
                    }
                }
            }
            Item::Enum(enumeration) => {
                for variant in &enumeration.variants {
                    if let Some(parameter) = untyped_parameter(&variant.fields) {
                        return Some((parameter.name.span, &parameter.name.name, "variant field"));
                    }
                }
            }
            Item::Import(_) | Item::Statement(_) => {}
        }
    }
    None
}

fn untyped_function_parameter(function: &FunctionDecl) -> Option<(Span, &str, &'static str)> {
    untyped_parameter(&function.params).map(|parameter| {
        (
            parameter.name.span,
            parameter.name.name.as_str(),
            "parameter",
        )
    })
}

fn untyped_parameter(parameters: &[Parameter]) -> Option<&Parameter> {
    parameters.iter().find(|parameter| parameter.ty.is_none())
}

fn source_snippet(source: &str, span: Span) -> String {
    let start = source[..span.start.min(source.len())]
        .rfind('\n')
        .map_or(0, |index| index + 1);
    let end = source[span.end.min(source.len())..]
        .find('\n')
        .map_or(source.len(), |index| span.end.min(source.len()) + index);
    source[start..end].trim().to_owned()
}

fn package_label(manifest_path: Option<&Path>) -> String {
    manifest_path
        .and_then(|path| parse_manifest(path).ok().map(|manifest| (path, manifest)))
        .and_then(|(path, manifest)| package_name(&manifest, path).ok().map(str::to_owned))
        .unwrap_or_else(|| "application".into())
}

fn enforce_manifest_unsafe_policy(
    manifest: &toml::Value,
    manifest_path: &Path,
    source_path: &Path,
    tokens: &[severian_lexer::Token],
    span: Span,
    interface_library: bool,
) -> Result<(), PackageError> {
    let package = package_name(manifest, manifest_path)?;
    let configuration = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("unsafe"))
        .and_then(toml::Value::as_table);
    let capabilities = unsafe_string_array(configuration, "capabilities", manifest_path)?;
    let sources = unsafe_string_array(configuration, "sources", manifest_path)?;
    let requested = unsafe_capabilities(tokens);
    let relative_source = manifest_relative_source(manifest_path, source_path);
    let source_allowed = sources
        .iter()
        .any(|source| Path::new(source) == relative_source);
    let is_library = interface_library || source_is_library(manifest, manifest_path, source_path);
    if requested.contains("native-abi") && !is_library {
        return Err(unsafe_policy_error(
            &format!("non-library target `{}`", relative_source.display()),
            span,
            "native-abi",
        ));
    }
    let missing = requested
        .iter()
        .find(|capability| !capabilities.iter().any(|allowed| allowed == *capability));
    if source_allowed && missing.is_none() {
        return Ok(());
    }
    let capability = missing.copied().unwrap_or("source-file");
    Err(unsafe_policy_error(
        &format!("package `{package}` source `{}`", relative_source.display()),
        span,
        capability,
    ))
}

fn unsafe_string_array(
    configuration: Option<&toml::Table>,
    key: &str,
    manifest_path: &Path,
) -> Result<Vec<String>, PackageError> {
    let Some(value) = configuration.and_then(|configuration| configuration.get(key)) else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "{}.package.unsafe.{key} must be an array of strings",
                manifest_path.display()
            ))
        })?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                PackageError::Manifest(format!(
                    "{}.package.unsafe.{key} must contain only strings",
                    manifest_path.display()
                ))
            })
        })
        .collect()
}

fn unsafe_capabilities(tokens: &[severian_lexer::Token]) -> BTreeSet<&'static str> {
    use severian_lexer::TokenKind;
    let mut capabilities = BTreeSet::new();
    if tokens.iter().any(|token| token.kind == TokenKind::Native) {
        capabilities.insert("native-abi");
    }
    if tokens
        .iter()
        .any(|token| token.kind == TokenKind::Ampersand)
    {
        capabilities.insert("pointers");
    }
    if tokens.iter().any(|token| token.kind == TokenKind::Async)
        && tokens
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "runtime"))
    {
        capabilities.insert("runtime-owned-tasks");
    }
    if capabilities.is_empty() {
        capabilities.insert("unsafe-blocks");
    }
    capabilities
}

fn manifest_relative_source(manifest_path: &Path, source_path: &Path) -> PathBuf {
    manifest_path
        .parent()
        .and_then(|root| source_path.strip_prefix(root).ok())
        .unwrap_or(source_path)
        .to_path_buf()
}

fn source_is_library(manifest: &toml::Value, manifest_path: &Path, source_path: &Path) -> bool {
    let Some(root) = manifest_path.parent() else {
        return false;
    };
    let configured = root.join(library_path(manifest));
    match (configured.canonicalize(), source_path.canonicalize()) {
        (Ok(configured), Ok(source)) => configured == source,
        _ => configured == source_path,
    }
}

fn unsafe_policy_error(reason: &str, span: Span, capability: &str) -> PackageError {
    PackageError::Frontend {
        package: reason.into(),
        stage: "unsafe policy",
        span,
        message: format!(
            "E0701: unsafe capability `{capability}` is not allowed; add it and this source path to `[package.unsafe]`, while native ABI remains library-only and tests remain safe-only"
        ),
        source_path: None,
        source: None,
    }
}

fn with_frontend_source(error: PackageError, path: &Path, source: &str) -> PackageError {
    match error {
        PackageError::Frontend {
            package,
            stage,
            span,
            message,
            ..
        } => PackageError::Frontend {
            package,
            stage,
            span,
            message,
            source_path: Some(path.to_path_buf()),
            source: Some(source.to_owned()),
        },
        error => error,
    }
}

pub fn load_path_dependency_interfaces(
    manifest_path: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    load_dependency_interfaces(resolve_dependencies(manifest_path)?)
}

pub fn load_transient_dependency_interfaces(
    manifest_path: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    load_dependency_interfaces(resolve_dependencies_transient(manifest_path)?)
}

fn load_dependency_interfaces(
    resolution: Resolution,
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut interfaces = Vec::new();
    for dependency in resolution.dependencies {
        interfaces.extend(load_interface_tree_as(
            &dependency.import_name,
            &dependency.package_name,
            &dependency.root,
        )?);
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

/// Loads quoted source imports from a project root. Quoted imports are never
/// resolved through the package registry or the official library directory.
pub fn load_local_interfaces(
    module: &Module,
    project_root: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let has_local_imports = module.items.iter().any(|item| {
        matches!(
            item,
            Item::Import(import) if matches!(import.kind, ImportKind::Local { .. })
        )
    });
    if !has_local_imports {
        return Ok(Vec::new());
    }
    let project_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    let project_root = project_root.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "local import root {} is invalid: {error}",
            project_root.display()
        ))
    })?;
    let mut visited = HashSet::new();
    let mut names = HashMap::new();
    let mut interfaces = Vec::new();
    collect_local_interfaces(
        module,
        &project_root,
        &mut visited,
        &mut names,
        &mut interfaces,
    )?;
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

fn collect_local_interfaces(
    module: &Module,
    project_root: &Path,
    visited: &mut HashSet<PathBuf>,
    names: &mut HashMap<String, PathBuf>,
    interfaces: &mut Vec<PackageInterface>,
) -> Result<(), PackageError> {
    for item in &module.items {
        let Item::Import(import) = item else {
            continue;
        };
        let ImportKind::Local { path, alias } = &import.kind else {
            continue;
        };
        let module_name = local_import_module_name(path).ok_or_else(|| {
            PackageError::Manifest(format!("local import path `{path}` is empty or invalid"))
        })?;
        if alias.is_none() && local_import_exposed_name(path).is_none() {
            return Err(PackageError::Manifest(format!(
                "local import `{path}` needs an identifier alias"
            )));
        }
        let source_path = resolve_local_import(project_root, path)?;
        if let Some(existing) = names.get(&module_name) {
            if existing != &source_path {
                return Err(PackageError::Manifest(format!(
                    "local imports {} and {} both resolve to module `{module_name}`",
                    existing.display(),
                    source_path.display()
                )));
            }
        } else {
            names.insert(module_name.clone(), source_path.clone());
        }
        if !visited.insert(source_path.clone()) {
            continue;
        }
        let source = std::fs::read_to_string(&source_path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read local import {}: {error}",
                source_path.display()
            ))
        })?;
        let tokens = severian_lexer::lex(&source).map_err(|error| PackageError::Frontend {
            package: module_name.clone(),
            stage: "lexer",
            span: error.span,
            message: error.message,
            source_path: Some(source_path.clone()),
            source: Some(source.clone()),
        })?;
        let imported = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
            package: module_name.clone(),
            stage: "parser",
            span: error.span,
            message: error.message,
            source_path: Some(source_path.clone()),
            source: Some(source.clone()),
        })?;
        collect_local_interfaces(&imported, project_root, visited, names, interfaces)?;
        interfaces.push(PackageInterface {
            name: module_name,
            export_package: None,
            module: imported,
            compiler: CompilerMetadata::default(),
            source_path,
            source,
        });
    }
    Ok(())
}

fn resolve_local_import(project_root: &Path, import: &str) -> Result<PathBuf, PackageError> {
    let relative = Path::new(import);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(PackageError::Manifest(format!(
            "local import `{import}` must stay within the project root"
        )));
    }
    let mut candidate = project_root.join(relative);
    if candidate.extension().is_none() {
        candidate.set_extension("sev");
    }
    let canonical = candidate.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "local import `{import}` does not resolve to {}: {error}",
            candidate.display()
        ))
    })?;
    if !canonical.starts_with(project_root) || !canonical.is_file() {
        return Err(PackageError::Manifest(format!(
            "local import `{import}` must resolve to a .sev file within {}",
            project_root.display()
        )));
    }
    Ok(canonical)
}

pub fn local_import_module_name(path: &str) -> Option<String> {
    let path = path.replace('\\', "/");
    let path = path.strip_suffix(".sev").unwrap_or(&path);
    let parts = path
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("."))
}

pub fn local_import_exposed_name(path: &str) -> Option<String> {
    let module = local_import_module_name(path)?;
    let name = module.rsplit('.').next()?;
    let mut bytes = name.bytes();
    let valid = bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric());
    valid.then(|| name.to_owned())
}

pub fn load_official_interfaces(
    module: &Module,
    library_root: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut pending = imported_packages(module)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut loaded = HashSet::new();
    let mut interfaces = Vec::new();
    while let Some(name) = pending.pop_first() {
        if !loaded.insert(name.clone()) {
            continue;
        }
        let Some((name, directory)) = ({
            let directory = name
                .split('.')
                .fold(library_root.to_path_buf(), |path, segment| {
                    path.join(segment)
                });
            manifest_in(&directory).map(|_| (name, directory))
        }) else {
            continue;
        };
        for interface in load_interface_tree_as(&name, &name, &directory)? {
            pending.extend(imported_packages(&interface.module));
            interfaces.push(interface);
        }
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

/// Loads standard packages embedded in the compiler binary. This keeps named
/// imports available when `sev` is installed or relocated without its source
/// checkout. An explicit `SEVERIAN_LIBRARY_PATH` can still select editable
/// on-disk packages during compiler and standard-library development.
pub fn load_embedded_official_interfaces(
    module: &Module,
    packages: &[EmbeddedOfficialPackage<'_>],
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut pending = imported_packages(module)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut loaded = HashSet::new();
    let mut interfaces = Vec::new();
    while let Some(name) = pending.pop_first() {
        if !loaded.insert(name.clone()) {
            continue;
        }
        let Some(package) = packages.iter().find(|package| package.name == name) else {
            continue;
        };
        let manifest_path = PathBuf::from("<severian-stdlib>")
            .join(package.name)
            .join(MANIFEST_FILE);
        let source_path = PathBuf::from("<severian-stdlib>")
            .join(package.name)
            .join("src/lib.sev");
        let manifest = toml::from_str::<toml::Value>(package.manifest).map_err(|error| {
            PackageError::Manifest(format!(
                "invalid embedded manifest for package `{}`: {error}",
                package.name
            ))
        })?;
        let declared_name = package_name(&manifest, &manifest_path)?;
        if declared_name != package.name {
            return Err(PackageError::Manifest(format!(
                "embedded package `{}` declares package `{declared_name}`",
                package.name
            )));
        }
        let interface = load_interface_source(
            &name,
            &manifest,
            &manifest_path,
            source_path,
            package.source.to_owned(),
        )?;
        pending.extend(imported_packages(&interface.module));
        interfaces.push(interface);
        for embedded in package.modules {
            let module_name = local_import_module_name(embedded.path).ok_or_else(|| {
                PackageError::Manifest(format!(
                    "embedded module path `{}` in package `{}` is invalid",
                    embedded.path, package.name
                ))
            })?;
            let module_path = PathBuf::from("<severian-stdlib>")
                .join(package.name)
                .join(embedded.path);
            let mut interface = load_interface_source(
                &module_name,
                &manifest,
                &manifest_path,
                module_path,
                embedded.source.to_owned(),
            )?;
            interface.export_package = Some(name.clone());
            pending.extend(imported_packages(&interface.module));
            interfaces.push(interface);
        }
    }
    interfaces.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(interfaces)
}

fn imported_packages(module: &Module) -> HashSet<String> {
    module
        .items
        .iter()
        .filter_map(|item| {
            let Item::Import(import) = item else {
                return None;
            };
            Some(match &import.kind {
                ImportKind::Local { .. } => Vec::new(),
                ImportKind::Module { path, .. } => vec![path
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".")],
                ImportKind::From { module, names } => {
                    let module = module
                        .iter()
                        .map(|part| part.name.as_str())
                        .collect::<Vec<_>>()
                        .join(".");
                    let mut packages = vec![module.clone()];
                    packages.extend(
                        names
                            .iter()
                            .map(|name| format!("{module}.{}", name.name.name)),
                    );
                    packages
                }
            })
        })
        .flatten()
        .collect()
}

fn load_interface_tree_as(
    import_name: &str,
    expected_package: &str,
    directory: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let root = load_interface_as(import_name, expected_package, directory)?;
    let mut local = load_local_interfaces(&root.module, directory)?;
    for interface in &mut local {
        interface.export_package = Some(import_name.to_owned());
    }
    let mut interfaces = Vec::with_capacity(local.len() + 1);
    interfaces.push(root);
    interfaces.extend(local);
    Ok(interfaces)
}

fn load_interface_as(
    import_name: &str,
    expected_package: &str,
    directory: &Path,
) -> Result<PackageInterface, PackageError> {
    let manifest_path = manifest_in(directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
    let manifest = parse_manifest(&manifest_path)?;
    let source_path = directory.join(library_path(&manifest));
    let source = std::fs::read_to_string(&source_path).map_err(|error| {
        PackageError::Manifest(format!(
            "could not read package `{expected_package}` at {}: {error}",
            source_path.display()
        ))
    })?;
    let declared_name = package_name(&manifest, &manifest_path)?;
    if declared_name != expected_package {
        return Err(PackageError::Manifest(format!(
            "dependency `{import_name}` expects package `{expected_package}` but resolves to `{declared_name}`"
        )));
    }
    load_interface_source(import_name, &manifest, &manifest_path, source_path, source)
}

fn load_interface_source(
    name: &str,
    manifest: &toml::Value,
    manifest_path: &Path,
    source_path: PathBuf,
    source: String,
) -> Result<PackageInterface, PackageError> {
    let tokens = severian_lexer::lex(&source).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "lexer",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.clone()),
        source: Some(source.clone()),
    })?;
    if let Some(token) = tokens
        .iter()
        .find(|token| token.kind == severian_lexer::TokenKind::Unsafe)
    {
        enforce_manifest_unsafe_policy(
            manifest,
            manifest_path,
            &source_path,
            &tokens,
            token.span,
            true,
        )
        .map_err(|error| with_frontend_source(error, &source_path, &source))?;
    }
    let module = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "parser",
        span: error.span,
        message: error.message,
        source_path: Some(source_path.clone()),
        source: Some(source.clone()),
    })?;
    Ok(PackageInterface {
        name: name.into(),
        export_package: None,
        module,
        compiler: compiler_metadata(name, manifest, manifest_path)?,
        source_path,
        source,
    })
}

fn compiler_metadata(
    package: &str,
    manifest: &toml::Value,
    manifest_path: &Path,
) -> Result<CompilerMetadata, PackageError> {
    let Some(compiler) = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("metadata"))
        .and_then(toml::Value::as_table)
        .and_then(|metadata| metadata.get("compiler"))
        .and_then(toml::Value::as_table)
    else {
        return Ok(CompilerMetadata::default());
    };

    let symbols = compiler
        .get("symbols")
        .and_then(toml::Value::as_table)
        .map(|symbols| {
            symbols
                .iter()
                .map(|(symbol, function)| {
                    function
                        .as_str()
                        .map(|function| (symbol.clone(), function.to_owned()))
                        .ok_or_else(|| {
                            metadata_error(
                                manifest_path,
                                format!("symbol `{symbol}` must name a function"),
                            )
                        })
                })
                .collect::<Result<HashMap<_, _>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let mut external_functions = compiler
        .get("external-functions")
        .and_then(toml::Value::as_array)
        .map(|functions| {
            functions
                .iter()
                .map(|function| {
                    function
                        .as_str()
                        .map(|function| format!("{package}.{function}"))
                        .ok_or_else(|| {
                            metadata_error(manifest_path, "external-functions must be strings")
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    external_functions.sort();
    external_functions.dedup();

    let mut fusion_aliases = compiler
        .get("fusion-aliases")
        .and_then(toml::Value::as_table)
        .map(|aliases| {
            aliases
                .iter()
                .map(|(function, target)| {
                    let target = target.as_str().ok_or_else(|| {
                        metadata_error(
                            manifest_path,
                            format!("fusion alias `{function}` must name a target function"),
                        )
                    })?;
                    Ok(FusionAlias {
                        function: format!("{package}.{function}"),
                        target: if target.contains('.') {
                            target.to_owned()
                        } else {
                            format!("{package}.{target}")
                        },
                    })
                })
                .collect::<Result<Vec<_>, PackageError>>()
        })
        .transpose()?
        .unwrap_or_default();
    fusion_aliases.sort_by(|left, right| left.function.cmp(&right.function));

    let mut fusion_rules = Vec::new();
    if let Some(fusion) = compiler.get("fusion").and_then(toml::Value::as_table) {
        let runtime_symbol = required_string(fusion, "runtime-symbol", manifest_path)?;
        if !valid_symbol(&runtime_symbol) {
            return Err(metadata_error(
                manifest_path,
                "fusion runtime-symbol is not a valid native symbol",
            ));
        }
        let packing_bits = optional_integer(fusion, "packing-bits", 4, manifest_path)?;
        let max_chain = optional_integer(fusion, "max-chain", 16, manifest_path)?;
        if !(1..=8).contains(&packing_bits) || max_chain == 0 || max_chain > 64 / packing_bits {
            return Err(metadata_error(
                manifest_path,
                "fusion packing-bits must be 1..=8 and its max-chain must fit in 64 bits",
            ));
        }
        let operations = fusion
            .get("operations")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| metadata_error(manifest_path, "fusion.operations is required"))?;
        for (function, opcode) in operations {
            let opcode = opcode.as_integer().ok_or_else(|| {
                metadata_error(
                    manifest_path,
                    format!("fusion opcode for `{function}` must be an integer"),
                )
            })?;
            let maximum_opcode = (1u16 << packing_bits) - 1;
            if opcode <= 0 || opcode > i64::from(maximum_opcode) {
                return Err(metadata_error(
                    manifest_path,
                    format!("fusion opcode for `{function}` does not fit packing-bits"),
                ));
            }
            fusion_rules.push(FusionRule {
                function: format!("{package}.{function}"),
                runtime_symbol: runtime_symbol.clone(),
                opcode: opcode as u8,
                packing_bits: packing_bits as u8,
                max_chain: max_chain as usize,
            });
        }
    }
    fusion_rules.sort_by(|left, right| left.function.cmp(&right.function));

    let mut graph_rules = compiler
        .get("graph-operations")
        .and_then(toml::Value::as_table)
        .map(|operations| {
            operations
                .iter()
                .map(|(function, operation)| {
                    let operation = operation.as_str().ok_or_else(|| {
                        metadata_error(
                            manifest_path,
                            format!("graph operation `{function}` must name an operation kind"),
                        )
                    })?;
                    let operation = match operation {
                        "input" => GraphOperation::Input,
                        "relu" => GraphOperation::Relu,
                        "add" => GraphOperation::Add,
                        "matmul" => GraphOperation::Matmul,
                        "transpose" => GraphOperation::Transpose,
                        "scale" => GraphOperation::Scale,
                        "softmax-rows" => GraphOperation::SoftmaxRows,
                        "layer-norm" => GraphOperation::LayerNorm,
                        "run" => GraphOperation::Run,
                        other => {
                            return Err(metadata_error(
                                manifest_path,
                                format!("unknown graph operation kind `{other}`"),
                            ));
                        }
                    };
                    Ok(GraphRule {
                        function: format!("{package}.{function}"),
                        operation,
                    })
                })
                .collect::<Result<Vec<_>, PackageError>>()
        })
        .transpose()?
        .unwrap_or_default();
    graph_rules.sort_by(|left, right| left.function.cmp(&right.function));
    Ok(CompilerMetadata {
        symbols,
        external_functions,
        fusion_rules,
        fusion_aliases,
        graph_rules,
    })
}

fn valid_symbol(symbol: &str) -> bool {
    let mut characters = symbol.chars();
    characters
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| {
            character == '_' || character == '.' || character.is_ascii_alphanumeric()
        })
}

fn required_string(table: &toml::Table, key: &str, path: &Path) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| metadata_error(path, format!("compiler fusion requires `{key}`")))
}

fn optional_integer(
    table: &toml::Table,
    key: &str,
    default: i64,
    path: &Path,
) -> Result<i64, PackageError> {
    match table.get(key) {
        Some(value) => value
            .as_integer()
            .ok_or_else(|| metadata_error(path, format!("`{key}` must be an integer"))),
        None => Ok(default),
    }
}

fn metadata_error(path: &Path, message: impl Into<String>) -> PackageError {
    PackageError::Manifest(format!(
        "invalid compiler metadata in {}: {}",
        path.display(),
        message.into()
    ))
}

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
