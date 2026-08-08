use crate::{
    dependency::{Dependency, DependencySource},
    package::{Package, PackageTarget, PackageTargetKind},
    profile::BuildProfile,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    MissingPackageName,
    InvalidDependency(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Parse(error) => error.fmt(formatter),
            Self::MissingPackageName => formatter.write_str("manifest has no [package].name"),
            Self::InvalidDependency(name) => write!(formatter, "invalid dependency `{name}`"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl From<std::io::Error> for ManifestError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<toml::de::Error> for ManifestError {
    fn from(error: toml::de::Error) -> Self {
        Self::Parse(error)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Manifest {
    pub package: Option<PackageSection>,
    pub workspace: Option<WorkspaceSection>,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DependencySpec>,
    #[serde(default)]
    pub features: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub profile: BTreeMap<String, ProfileSection>,
    #[serde(default)]
    pub lints: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub build: BuildSection,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageSection {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub edition: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WorkspaceSection {
    #[serde(default)]
    pub members: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub default_members: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BuildSection {
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub target_directory: Option<PathBuf>,
    #[serde(default)]
    pub jobs: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProfileSection {
    #[serde(default)]
    pub inherits: Option<String>,
    #[serde(default)]
    pub optimization: Option<u8>,
    #[serde(default)]
    pub debug: Option<toml::Value>,
    #[serde(default)]
    pub lto: Option<toml::Value>,
    #[serde(default)]
    pub incremental: Option<bool>,
    #[serde(default)]
    pub overflow_checks: Option<bool>,
    #[serde(default)]
    pub assertions: Option<bool>,
    #[serde(default)]
    pub runtime_checks: Option<bool>,
    #[serde(default)]
    pub coverage: Option<bool>,
    #[serde(default)]
    pub sanitizer: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Version(String),
    Detailed(DependencyDetail),
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DependencyDetail {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub git: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub rev: Option<String>,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub package: Option<String>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default = "default_true")]
    pub default_features: bool,
}

impl Manifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let text = fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    pub fn to_package(&self, root: impl Into<PathBuf>) -> Result<Package, ManifestError> {
        let root = root.into();
        let package = self.package.as_ref().ok_or(ManifestError::MissingPackageName)?;
        let dependencies = self
            .dependencies
            .iter()
            .map(|(name, spec)| dependency_from_spec(name, spec))
            .collect::<Result<Vec<_>, _>>()?;

        let mut targets = Vec::new();
        let main = root.join("src/main.sev");
        let library = root.join("src/lib.sev");
        if main.exists() {
            targets.push(PackageTarget {
                name: package.name.clone(),
                kind: PackageTargetKind::Binary,
                source: main,
            });
        }
        if library.exists() {
            targets.push(PackageTarget {
                name: package.name.clone(),
                kind: PackageTargetKind::Library,
                source: library,
            });
        }

        Ok(Package {
            name: package.name.clone(),
            version: package.version.clone(),
            root,
            targets,
            dependencies,
        })
    }

    pub fn profiles(&self) -> Result<BTreeMap<String, BuildProfile>, String> {
        BuildProfile::resolve_all(&self.profile)
    }
}

fn dependency_from_spec(
    name: &str,
    specification: &DependencySpec,
) -> Result<Dependency, ManifestError> {
    match specification {
        DependencySpec::Version(version) => Ok(Dependency::registry(name, version)),
        DependencySpec::Detailed(detail) => {
            let source = if let Some(path) = &detail.path {
                DependencySource::Path(path.clone())
            } else if let Some(url) = &detail.git {
                DependencySource::Git {
                    url: url.clone(),
                    revision: detail.rev.clone(),
                    branch: detail.branch.clone(),
                    tag: detail.tag.clone(),
                }
            } else if let Some(version) = &detail.version {
                DependencySource::Registry {
                    registry: detail.registry.clone(),
                    version: version.clone(),
                }
            } else {
                return Err(ManifestError::InvalidDependency(name.to_owned()));
            };

            Ok(Dependency {
                name: name.to_owned(),
                package: detail.package.clone(),
                source,
                optional: detail.optional,
                features: detail.features.clone(),
                default_features: detail.default_features,
            })
        }
    }
}

fn default_version() -> String {
    "0.1.0".into()
}

const fn default_true() -> bool {
    true
}
