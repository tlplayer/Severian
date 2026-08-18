use crate::PackageError;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemRequirement {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallRequirement {
    pub name: String,
    pub publisher: String,
    pub package: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallationManifest {
    pub package_name: String,
    pub package_version: String,
    pub root: PathBuf,
    pub system: Vec<SystemRequirement>,
    pub install: Vec<InstallRequirement>,
}

pub fn read_installation_manifest(path: &Path) -> Result<InstallationManifest, PackageError> {
    let source = fs::read_to_string(path)?;
    let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
        PackageError::Manifest(format!("invalid manifest {}: {error}", path.display()))
    })?;
    validate_non_executable_manifest(&value, path)?;
    let package = value
        .get("package")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| manifest_error(path, "missing `[package]` table"))?;
    let package_name = required_string(package, "name", path, "package")?;
    let package_version = required_string(package, "version", path, "package")?;
    let system = parse_system(&value, path)?;
    let install = parse_install(&value, path)?;
    let system_names = system
        .iter()
        .map(|item| item.name.as_str())
        .collect::<BTreeSet<_>>();
    for item in &install {
        if !system_names.contains(item.name.as_str()) {
            return Err(manifest_error(
                path,
                format!(
                    "`[install.{}]` is undeclared; add `{}` to `[system]`",
                    item.name, item.name
                ),
            ));
        }
    }
    Ok(InstallationManifest {
        package_name,
        package_version,
        root: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        system,
        install,
    })
}

fn parse_system(value: &toml::Value, path: &Path) -> Result<Vec<SystemRequirement>, PackageError> {
    let Some(table) = value.get("system") else {
        return Ok(Vec::new());
    };
    table
        .as_table()
        .ok_or_else(|| manifest_error(path, "`[system]` must be a table"))?
        .iter()
        .map(|(name, requirement)| {
            let version = requirement.as_str().ok_or_else(|| {
                manifest_error(path, format!("`system.{name}` must be a version string"))
            })?;
            validate_identifier(name, path, "system requirement")?;
            if version.trim().is_empty() {
                return Err(manifest_error(
                    path,
                    format!("`system.{name}` cannot be empty"),
                ));
            }
            Ok(SystemRequirement {
                name: name.clone(),
                version: version.to_owned(),
            })
        })
        .collect()
}

fn parse_install(
    value: &toml::Value,
    path: &Path,
) -> Result<Vec<InstallRequirement>, PackageError> {
    let Some(table) = value.get("install") else {
        return Ok(Vec::new());
    };
    table
        .as_table()
        .ok_or_else(|| manifest_error(path, "`[install]` must contain named requirement tables"))?
        .iter()
        .map(|(name, value)| {
            validate_identifier(name, path, "install requirement")?;
            let detail = value.as_table().ok_or_else(|| {
                manifest_error(path, format!("`[install.{name}]` must be a table"))
            })?;
            let allowed = BTreeSet::from(["publisher", "package", "source"]);
            for key in detail.keys() {
                if !allowed.contains(key.as_str()) {
                    return Err(manifest_error(
                        path,
                        format!(
                            "`install.{name}.{key}` is not allowed; package manifests may declare intent but cannot provide installer code"
                        ),
                    ));
                }
            }
            let source = required_string(detail, "source", path, &format!("install.{name}"))?;
            if source != "vendor" {
                return Err(manifest_error(
                    path,
                    format!(
                        "`install.{name}.source` must be `vendor`; packages cannot request arbitrary URLs"
                    ),
                ));
            }
            let publisher = required_string(detail, "publisher", path, &format!("install.{name}"))?;
            let package = required_string(detail, "package", path, &format!("install.{name}"))?;
            validate_identifier(&publisher, path, "publisher")?;
            validate_identifier(&package, path, "vendor package")?;
            Ok(InstallRequirement {
                name: name.clone(),
                publisher,
                package,
                source,
            })
        })
        .collect()
}

pub(crate) fn validate_non_executable_manifest(
    value: &toml::Value,
    path: &Path,
) -> Result<(), PackageError> {
    const FORBIDDEN: &[&str] = &[
        "install_script",
        "install-script",
        "setup",
        "setup_py",
        "setup-py",
        "shell",
        "command",
        "powershell",
        "postinstall",
        "preinstall",
        "build_script",
        "build-script",
    ];
    fn walk(value: &toml::Value, prefix: &str, path: &Path) -> Result<(), PackageError> {
        match value {
            toml::Value::Table(table) => {
                for (key, value) in table {
                    let full = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    if FORBIDDEN.contains(&key.as_str()) {
                        return Err(manifest_error(
                            path,
                            format!("`{full}` is an executable installer hook; package.toml is declarative only"),
                        ));
                    }
                    walk(value, &full, path)?;
                }
            }
            toml::Value::Array(values) => {
                for value in values {
                    walk(value, prefix, path)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(value, "", path)
}

pub(crate) fn system_requirements_by_name(manifest: &InstallationManifest) -> BTreeMap<&str, &str> {
    manifest
        .system
        .iter()
        .map(|item| (item.name.as_str(), item.version.as_str()))
        .collect()
}

fn required_string(
    table: &toml::Table,
    key: &str,
    path: &Path,
    context: &str,
) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            manifest_error(
                path,
                format!("`{context}.{key}` must be a non-empty string"),
            )
        })
}

fn validate_identifier(name: &str, path: &Path, context: &str) -> Result<(), PackageError> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(manifest_error(
            path,
            format!("invalid {context} name `{name}`"),
        ));
    }
    Ok(())
}

fn manifest_error(path: &Path, message: impl std::fmt::Display) -> PackageError {
    PackageError::Manifest(format!("{}: {message}", path.display()))
}
