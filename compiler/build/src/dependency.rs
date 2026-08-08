use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencySource {
    Registry {
        registry: Option<String>,
        version: String,
    },
    Path(PathBuf),
    Git {
        url: String,
        revision: Option<String>,
        branch: Option<String>,
        tag: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub name: String,
    pub package: Option<String>,
    pub source: DependencySource,
    pub optional: bool,
    pub features: Vec<String>,
    pub default_features: bool,
}

impl Dependency {
    pub fn registry(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            package: None,
            source: DependencySource::Registry {
                registry: None,
                version: version.into(),
            },
            optional: false,
            features: Vec::new(),
            default_features: true,
        }
    }
}
