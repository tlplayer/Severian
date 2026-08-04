#![forbid(unsafe_code)]

use severian_ast::{ImportKind, Item, Module, Span};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompilerMetadata {
    pub symbols: HashMap<String, String>,
    pub external_functions: Vec<String>,
    pub fusion_rules: Vec<FusionRule>,
    pub fusion_aliases: Vec<FusionAlias>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackageInterface {
    pub name: String,
    pub module: Module,
    pub compiler: CompilerMetadata,
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
    source
        .parent()?
        .ancestors()
        .map(|directory| directory.join("Severian.toml"))
        .find(|candidate| candidate.is_file())
}

/// Resolves the first binary target for `sev build` from the nearest manifest.
pub fn default_binary_source(directory: &Path) -> Result<PathBuf, PackageError> {
    let direct = directory.join("main.sev");
    if direct.is_file() {
        return Ok(direct);
    }
    let manifest_path = directory
        .ancestors()
        .map(|ancestor| ancestor.join("Severian.toml"))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "could not find `main.sev` or Severian.toml from {}",
                directory.display()
            ))
        })?;
    let manifest = parse_manifest(&manifest_path)?;
    let binary_path = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .and_then(|binaries| binaries.first())
        .and_then(toml::Value::as_table)
        .and_then(|binary| binary.get("path"))
        .and_then(toml::Value::as_str)
        .unwrap_or("src/main.sev");
    let source = manifest_path
        .parent()
        .expect("a manifest path has a parent")
        .join(binary_path);
    if !source.is_file() {
        return Err(PackageError::Manifest(format!(
            "binary source {} does not exist",
            source.display()
        )));
    }
    Ok(source)
}

pub fn load_path_dependency_sources(manifest_path: &Path) -> Result<Vec<String>, PackageError> {
    let mut visited = HashSet::new();
    let mut sources = Vec::new();
    load_manifest_dependencies(manifest_path, &mut visited, &mut sources)?;
    Ok(sources)
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
            let directory = library_root.join(&name);
            directory
                .join("Severian.toml")
                .is_file()
                .then_some((name, directory))
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
            let path = match &import.kind {
                ImportKind::Module { path, .. } => path,
                ImportKind::From { module, .. } => module,
            };
            path.first().map(|root| root.name.clone())
        })
        .collect()
}

fn load_interface(name: &str, directory: &Path) -> Result<PackageInterface, PackageError> {
    let manifest_path = directory.join("Severian.toml");
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
    Ok(CompilerMetadata {
        symbols,
        external_functions,
        fusion_rules,
        fusion_aliases,
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
        let dependency_manifest = dependency_directory.join("Severian.toml");
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
        sources.push(std::fs::read_to_string(&source_path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read library for `{dependency_name}` at {}: {error}",
                source_path.display()
            ))
        })?);
    }
    Ok(())
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
