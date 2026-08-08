use crate::fingerprint::Fingerprint;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuildNodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildStage {
    Parse,
    Semantic,
    Ownership,
    Check,
    Optimize,
    Lower,
    Codegen,
    Link,
    Test,
    Bench,
    Coverage,
    Custom,
}

impl BuildStage {
    pub const fn cacheable(self) -> bool {
        !matches!(self, Self::Test | Self::Bench | Self::Coverage)
    }
}

#[derive(Debug, Clone)]
pub struct BuildNode {
    pub id: BuildNodeId,
    pub package: String,
    pub target: String,
    pub stage: BuildStage,
    pub source_files: Vec<PathBuf>,
    pub dependencies: Vec<BuildNodeId>,
    pub outputs: Vec<PathBuf>,
    pub fingerprint: Option<Fingerprint>,
}

impl BuildNode {
    pub fn new(
        id: BuildNodeId,
        package: impl Into<String>,
        target: impl Into<String>,
        stage: BuildStage,
    ) -> Self {
        Self {
            id,
            package: package.into(),
            target: target.into(),
            stage,
            source_files: Vec::new(),
            dependencies: Vec::new(),
            outputs: Vec::new(),
            fingerprint: None,
        }
    }

    pub fn label(&self) -> String {
        format!("{}:{}:{:?}", self.package, self.target, self.stage)
    }

    pub fn depends_on(mut self, dependency: BuildNodeId) -> Self {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
        }
        self
    }

    pub fn with_source(mut self, source: impl Into<PathBuf>) -> Self {
        self.source_files.push(source.into());
        self
    }

    pub fn with_output(mut self, output: impl Into<PathBuf>) -> Self {
        self.outputs.push(output.into());
        self
    }
}
