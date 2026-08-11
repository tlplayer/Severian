#![forbid(unsafe_code)]

use severian_ast::{ImportKind, Item, Module, Span};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

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
    pub module: Module,
    pub compiler: CompilerMetadata,
    pub source_path: PathBuf,
    pub source: String,
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
            } => write!(
                formatter,
                "invalid {stage} in package `{package}` at bytes {}..{}: {message}",
                span.start, span.end
            ),
        }
    }
}

impl std::error::Error for PackageError {}

impl From<std::io::Error> for PackageError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
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
    let mut visited = HashSet::new();
    let mut targets = Vec::new();
    collect_library_targets(manifest_path, &mut visited, &mut targets)?;
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
    let mut visited = HashSet::new();
    let mut sources = Vec::new();
    load_manifest_dependencies(manifest_path, &mut visited, &mut sources)?;
    Ok(sources)
}

pub fn load_path_dependency_interfaces(
    manifest_path: &Path,
) -> Result<Vec<PackageInterface>, PackageError> {
    let mut visited = HashSet::new();
    let mut interfaces = Vec::new();
    collect_path_dependency_interfaces(manifest_path, &mut visited, &mut interfaces)?;
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
        })?;
        let imported = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
            package: module_name.clone(),
            stage: "parser",
            span: error.span,
            message: error.message,
        })?;
        collect_local_interfaces(&imported, project_root, visited, names, interfaces)?;
        interfaces.push(PackageInterface {
            name: module_name,
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

fn collect_path_dependency_interfaces(
    manifest_path: &Path,
    visited: &mut HashSet<PathBuf>,
    interfaces: &mut Vec<PackageInterface>,
) -> Result<(), PackageError> {
    let manifest = parse_manifest(manifest_path)?;
    let Some(dependencies) = manifest.get("dependencies").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    let root = manifest_path
        .parent()
        .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
    for (dependency_name, dependency) in dependencies {
        let Some(path) = dependency
            .as_table()
            .and_then(|table| table.get("path"))
            .and_then(toml::Value::as_str)
        else {
            continue;
        };
        let directory = root.join(path).canonicalize().map_err(|error| {
            PackageError::Manifest(format!(
                "dependency `{dependency_name}` has invalid path `{}`: {error}",
                root.join(path).display()
            ))
        })?;
        let dependency_manifest =
            manifest_in(&directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
        let canonical_manifest = dependency_manifest.canonicalize()?;
        if !visited.insert(canonical_manifest.clone()) {
            continue;
        }
        let dependency = parse_manifest(&canonical_manifest)?;
        let declared_name = package_name(&dependency, &canonical_manifest)?;
        if declared_name != dependency_name {
            return Err(PackageError::Manifest(format!(
                "dependency `{dependency_name}` resolves to package `{declared_name}`"
            )));
        }
        collect_path_dependency_interfaces(&canonical_manifest, visited, interfaces)?;
        interfaces.push(load_interface(dependency_name, &directory)?);
    }
    Ok(())
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
        let interface = load_interface(&name, &directory)?;
        pending.extend(imported_packages(&interface.module));
        interfaces.push(interface);
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

fn load_interface(name: &str, directory: &Path) -> Result<PackageInterface, PackageError> {
    let manifest_path = manifest_in(directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
    let manifest = parse_manifest(&manifest_path)?;
    let declared_name = package_name(&manifest, &manifest_path)?;
    if declared_name != name {
        return Err(PackageError::Manifest(format!(
            "official package `{name}` declares package `{declared_name}`"
        )));
    }
    let source_path = directory.join(library_path(&manifest));
    let source = std::fs::read_to_string(&source_path).map_err(|error| {
        PackageError::Manifest(format!(
            "could not read official package `{name}` at {}: {error}",
            source_path.display()
        ))
    })?;
    let tokens = severian_lexer::lex(&source).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "lexer",
        span: error.span,
        message: error.message,
    })?;
    let module = severian_parser::parse(&tokens).map_err(|error| PackageError::Frontend {
        package: name.into(),
        stage: "parser",
        span: error.span,
        message: error.message,
    })?;
    Ok(PackageInterface {
        name: name.into(),
        module,
        compiler: compiler_metadata(name, &manifest, &manifest_path)?,
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

fn load_manifest_dependencies(
    manifest_path: &Path,
    visited: &mut HashSet<PathBuf>,
    sources: &mut Vec<String>,
) -> Result<(), PackageError> {
    let manifest = parse_manifest(manifest_path)?;
    let Some(dependencies) = manifest.get("dependencies").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    let manifest_directory = manifest_path
        .parent()
        .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
    for (dependency_name, dependency) in dependencies {
        let Some(path) = dependency
            .as_table()
            .and_then(|table| table.get("path"))
            .and_then(toml::Value::as_str)
        else {
            continue;
        };
        let dependency_directory = manifest_directory.join(path);
        let dependency_manifest = manifest_in(&dependency_directory)
            .unwrap_or_else(|| dependency_directory.join(MANIFEST_FILE));
        let canonical_manifest = dependency_manifest.canonicalize().map_err(|error| {
            PackageError::Manifest(format!(
                "dependency `{dependency_name}` has invalid path `{}`: {error}",
                dependency_directory.display()
            ))
        })?;
        if !visited.insert(canonical_manifest.clone()) {
            continue;
        }
        let dependency_package = parse_manifest(&canonical_manifest)?;
        let declared_name = package_name(&dependency_package, &canonical_manifest)?;
        if declared_name != dependency_name {
            return Err(PackageError::Manifest(format!(
                "dependency `{dependency_name}` resolves to package `{declared_name}`"
            )));
        }
        load_manifest_dependencies(&canonical_manifest, visited, sources)?;
        let source_path = canonical_manifest
            .parent()
            .ok_or_else(|| PackageError::Manifest("dependency manifest has no parent".into()))?
            .join(library_path(&dependency_package));
        let artifact_path = library_artifact_path(
            canonical_manifest
                .parent()
                .expect("a manifest path has a parent"),
            declared_name,
        );
        let selected_path = if artifact_is_fresh(&artifact_path, &source_path, &canonical_manifest)
        {
            &artifact_path
        } else {
            &source_path
        };
        sources.push(std::fs::read_to_string(selected_path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read library for `{dependency_name}` at {}: {error}",
                selected_path.display()
            ))
        })?);
    }
    Ok(())
}

fn collect_library_targets(
    manifest_path: &Path,
    visited: &mut HashSet<PathBuf>,
    targets: &mut Vec<LibraryTarget>,
) -> Result<(), PackageError> {
    let canonical_manifest = manifest_path.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "package manifest {} is invalid: {error}",
            manifest_path.display()
        ))
    })?;
    if !visited.insert(canonical_manifest.clone()) {
        return Ok(());
    }
    let manifest = parse_manifest(&canonical_manifest)?;
    let directory = canonical_manifest
        .parent()
        .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
    if let Some(dependencies) = manifest.get("dependencies").and_then(toml::Value::as_table) {
        for dependency in dependencies.values() {
            let Some(path) = dependency
                .as_table()
                .and_then(|table| table.get("path"))
                .and_then(toml::Value::as_str)
            else {
                continue;
            };
            let dependency_directory = directory.join(path);
            let dependency_manifest = manifest_in(&dependency_directory)
                .unwrap_or_else(|| dependency_directory.join(MANIFEST_FILE));
            collect_library_targets(&dependency_manifest, visited, targets)?;
        }
    }
    if manifest.get("lib").is_some() {
        let name = package_name(&manifest, &canonical_manifest)?.to_owned();
        targets.push(LibraryTarget {
            source: directory.join(library_path(&manifest)),
            artifact: library_artifact_path(directory, &name),
            name,
            manifest: canonical_manifest,
        });
    }
    Ok(())
}

fn library_artifact_path(directory: &Path, package: &str) -> PathBuf {
    directory
        .join("target")
        .join("debug")
        .join("deps")
        .join(format!("lib{package}.sevi"))
}

fn artifact_is_fresh(artifact: &Path, source: &Path, manifest: &Path) -> bool {
    let Ok(artifact_modified) = artifact.metadata().and_then(|metadata| metadata.modified()) else {
        return false;
    };
    [source, manifest].iter().all(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| modified <= artifact_modified)
    })
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
    toml::from_str::<toml::Value>(&source).map_err(|error| {
        PackageError::Manifest(format!("invalid manifest {}: {error}", path.display()))
    })
}

fn manifest_in(directory: &Path) -> Option<PathBuf> {
    let manifest = directory.join(MANIFEST_FILE);
    manifest.is_file().then_some(manifest)
}
