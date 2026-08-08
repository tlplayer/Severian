use crate::dependency::Dependency;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageTargetKind {
    Library,
    Binary,
    Test,
    Benchmark,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageTarget {
    pub name: String,
    pub kind: PackageTargetKind,
    pub source: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub root: PathBuf,
    pub targets: Vec<PackageTarget>,
    pub dependencies: Vec<Dependency>,
}

impl Package {
    pub fn primary_target(&self) -> Option<&PackageTarget> {
        self.targets
            .iter()
            .find(|target| target.kind == PackageTargetKind::Binary)
            .or_else(|| {
                self.targets
                    .iter()
                    .find(|target| target.kind == PackageTargetKind::Library)
            })
    }
}
