use crate::manifest::{system_requirements_by_name, InstallationManifest};
use crate::signature::{validate_sha256, verify_ed25519};
use crate::trust::{severian_home, validate_publisher, Date, TrustRegistry};
use crate::{LockedExternal, PackageError};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VendorPackage {
    pub name: String,
    pub version: String,
    pub publisher: String,
    pub source: String,
    pub sha256: String,
    pub signature: String,
    pub artifact: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallPlanItem {
    pub locked: LockedExternal,
    pub requested_by: String,
    pub system_install: bool,
    pub artifact: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VendorCatalog {
    pub packages: Vec<VendorPackage>,
}

impl VendorCatalog {
    pub fn load_default() -> Result<Self, PackageError> {
        Self::load(&severian_home().join("trust/vendor-catalog.toml"))
    }

    pub fn load(path: &Path) -> Result<Self, PackageError> {
        let source = fs::read_to_string(path).map_err(|error| {
            PackageError::Manifest(format!(
                "could not read trusted vendor catalog {}: {error}",
                path.display()
            ))
        })?;
        let value = toml::from_str::<toml::Value>(&source).map_err(|error| {
            PackageError::Manifest(format!(
                "invalid vendor catalog {}: {error}",
                path.display()
            ))
        })?;
        let entries = value
            .get("package")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                PackageError::Manifest(format!("{} has no `[[package]]` entries", path.display()))
            })?;
        let root = path.parent().unwrap_or(Path::new("."));
        let packages = entries
            .iter()
            .map(|entry| parse_package(entry, path, root))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { packages })
    }
}

pub fn resolve_external_requirements(
    manifest: &InstallationManifest,
    trust: &TrustRegistry,
    catalog: &VendorCatalog,
    today: &Date,
) -> Result<Vec<InstallPlanItem>, PackageError> {
    let requirements = system_requirements_by_name(manifest);
    manifest
        .install
        .iter()
        .map(|request| {
            let version_requirement = requirements.get(request.name.as_str()).ok_or_else(|| {
                PackageError::Manifest(format!("install requirement `{}` is not declared in `[system]`", request.name))
            })?;
            let publisher = trust.publisher(&request.publisher)?;
            let mut candidates = catalog
                .packages
                .iter()
                .filter(|candidate| {
                    candidate.name == request.package
                        && candidate.publisher == request.publisher
                        && version_matches(version_requirement, &candidate.version)
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| compare_versions(&left.version, &right.version));
            let candidate = candidates.pop().ok_or_else(|| {
                PackageError::Manifest(format!(
                    "trusted vendor catalog has no `{}` version matching `{version_requirement}` from `{}`",
                    request.package, request.publisher
                ))
            })?;
            validate_sha256(&candidate.sha256)?;
            validate_publisher(publisher, &candidate.name, &candidate.source, today)?;
            verify_ed25519(
                &publisher.signing_keys,
                signature_payload(candidate).as_bytes(),
                &candidate.signature,
            )?;
            Ok(InstallPlanItem {
                locked: LockedExternal {
                    name: candidate.name.clone(),
                    version: candidate.version.clone(),
                    publisher: candidate.publisher.clone(),
                    source: candidate.source.clone(),
                    sha256: candidate.sha256.to_ascii_lowercase(),
                    signature: candidate.signature.to_ascii_lowercase(),
                    trusted_from: publisher.trusted_from.as_str().to_owned(),
                    trusted_until: publisher.trusted_until.as_str().to_owned(),
                },
                requested_by: manifest.package_name.clone(),
                system_install: publisher.allow_system_install,
                artifact: candidate.artifact.clone(),
            })
        })
        .collect()
}

pub fn signature_payload(package: &VendorPackage) -> String {
    format!(
        "name={}\nversion={}\npublisher={}\nsource={}\nsha256={}\n",
        package.name,
        package.version,
        package.publisher,
        package.source,
        package.sha256.to_ascii_lowercase()
    )
}

fn parse_package(
    value: &toml::Value,
    path: &Path,
    root: &Path,
) -> Result<VendorPackage, PackageError> {
    let table = value.as_table().ok_or_else(|| {
        PackageError::Manifest(format!(
            "package entry in {} must be a table",
            path.display()
        ))
    })?;
    let artifact = table
        .get("artifact")
        .and_then(toml::Value::as_str)
        .map(|artifact| {
            let artifact = PathBuf::from(artifact);
            if artifact.is_absolute() {
                artifact
            } else {
                root.join(artifact)
            }
        });
    Ok(VendorPackage {
        name: string(table, "name", path)?,
        version: string(table, "version", path)?,
        publisher: string(table, "publisher", path)?,
        source: string(table, "source", path)?,
        sha256: string(table, "sha256", path)?,
        signature: string(table, "signature", path)?,
        artifact,
    })
}

fn string(table: &toml::Table, key: &str, path: &Path) -> Result<String, PackageError> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "catalog package in {} has no string `{key}`",
                path.display()
            ))
        })
}

fn parse_version(value: &str) -> Option<[u64; 3]> {
    let core = value.split_once('-').map_or(value, |(core, _)| core);
    let parts = core
        .split('.')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    Some([
        *parts.first().unwrap_or(&0),
        *parts.get(1).unwrap_or(&0),
        *parts.get(2).unwrap_or(&0),
    ])
}

pub(crate) fn version_matches(requirement: &str, version: &str) -> bool {
    let Some(version) = parse_version(version) else {
        return false;
    };
    let requirement = requirement.trim();
    for operator in [">=", "<=", "==", ">", "<", "="] {
        if let Some(value) = requirement.strip_prefix(operator) {
            let Some(expected) = parse_version(value.trim()) else {
                return false;
            };
            return match operator {
                ">=" => version >= expected,
                "<=" => version <= expected,
                ">" => version > expected,
                "<" => version < expected,
                _ => version == expected,
            };
        }
    }
    let Some(expected) = parse_version(requirement) else {
        return false;
    };
    let components = requirement
        .split_once('-')
        .map_or(requirement, |(core, _)| core)
        .split('.')
        .count();
    match components {
        1 => version[0] == expected[0],
        2 => version[..2] == expected[..2],
        _ => version == expected,
    }
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    parse_version(left)
        .cmp(&parse_version(right))
        .then_with(|| left.cmp(right))
}
