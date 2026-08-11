use crate::{
    manifest::{Manifest, ManifestError},
    package::Package,
    DEFAULT_TARGET_DIRECTORY, MANIFEST_FILE,
};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub enum WorkspaceError {
    Manifest(ManifestError),
    Io(std::io::Error),
    MemberOutsideWorkspace(PathBuf),
}

impl std::fmt::Display for WorkspaceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manifest(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
            Self::MemberOutsideWorkspace(path) => write!(
                formatter,
                "workspace member resolves outside workspace root: {}",
                path.display()
            ),
        }
    }
}
impl std::error::Error for WorkspaceError {}

impl From<ManifestError> for WorkspaceError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}
impl From<std::io::Error> for WorkspaceError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
    pub root_manifest: Manifest,
    pub packages: Vec<Package>,
    pub target_directory: PathBuf,
}

impl Workspace {
    pub fn discover(start: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        let root = find_workspace_root(start.as_ref())?;
        let manifest_path = root.join(MANIFEST_FILE);
        let root_manifest = Manifest::load(&manifest_path)?;

        let target_directory = root_manifest
            .build
            .target_directory
            .clone()
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                }
            })
            .unwrap_or_else(|| root.join(DEFAULT_TARGET_DIRECTORY));

        let mut member_paths = BTreeSet::new();
        if let Some(workspace) = &root_manifest.workspace {
            for member in &workspace.members {
                member_paths.insert(root.join(member));
            }
        }
        if root_manifest.package.is_some() {
            member_paths.insert(root.clone());
        }

        let canonical_root = std::fs::canonicalize(&root)?;
        let mut packages = Vec::new();

        for member in member_paths {
            let canonical = std::fs::canonicalize(&member)?;
            if !canonical.starts_with(&canonical_root) {
                return Err(WorkspaceError::MemberOutsideWorkspace(member));
            }

            let manifest = if canonical == canonical_root {
                root_manifest.clone()
            } else {
                Manifest::load(canonical.join(MANIFEST_FILE))?
            };
            if manifest.package.is_some() {
                packages.push(manifest.to_package(canonical)?);
            }
        }

        packages.sort_by(|left, right| left.name.cmp(&right.name));

        Ok(Self {
            root,
            root_manifest,
            packages,
            target_directory,
        })
    }

    pub fn package(&self, name: &str) -> Option<&Package> {
        self.packages.iter().find(|package| package.name == name)
    }
}

fn find_workspace_root(start: &Path) -> Result<PathBuf, std::io::Error> {
    let mut current = if start.is_file() {
        start.parent().unwrap_or(start).to_path_buf()
    } else {
        start.to_path_buf()
    };

    loop {
        let manifest = current.join(MANIFEST_FILE);
        if manifest.is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("could not find {MANIFEST_FILE}"),
            ));
        }
    }
}
