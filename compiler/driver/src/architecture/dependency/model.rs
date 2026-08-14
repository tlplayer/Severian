use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureNode {
    pub name: String,
    pub path: PathBuf,
    pub manifest: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureDependency {
    pub source: usize,
    pub target: usize,
    pub manifest: PathBuf,
    pub line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureFinding {
    pub code: &'static str,
    pub severity: &'static str,
    pub manifest: PathBuf,
    pub line: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyStat {
    pub node: usize,
    pub fan_in: usize,
    pub fan_out: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DependencyAnalysis {
    pub nodes: Vec<ArchitectureNode>,
    pub dependencies: Vec<ArchitectureDependency>,
    pub findings: Vec<ArchitectureFinding>,
    pub stats: Vec<DependencyStat>,
}

impl DependencyAnalysis {
    pub fn to_dot(&self) -> String {
        let mut output = String::from("digraph severian_architecture {\n");
        for (index, node) in self.nodes.iter().enumerate() {
            let label = format!("{}\\n{}", node.name, node.path.display()).replace('"', "\\\"");
            output.push_str(&format!("  n{index} [label=\"{label}\"];\n"));
        }
        for dependency in &self.dependencies {
            output.push_str(&format!(
                "  n{} -> n{};\n",
                dependency.source, dependency.target
            ));
        }
        output.push_str("}\n");
        output
    }
}
