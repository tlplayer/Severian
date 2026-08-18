use severian_package::{architecture_path_matches, ArchitecturePolicy, BuildPolicy};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

mod graph;
mod model;

use graph::{adjacency, cycle_path, strongly_connected_components};
pub use model::{
    ArchitectureDependency, ArchitectureFinding, ArchitectureNode, DependencyAnalysis,
    DependencyStat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ManifestKind {
    Cargo,
    Severian,
}

impl ManifestKind {
    const fn file_name(self) -> &'static str {
        match self {
            Self::Cargo => "Cargo.toml",
            Self::Severian => "package.toml",
        }
    }
}

#[derive(Debug)]
struct LoadedNode {
    public: ArchitectureNode,
    kind: ManifestKind,
    source: String,
    value: toml::Value,
}

#[derive(Debug)]
struct DeclaredDependency {
    key: String,
    package: String,
    path: Option<PathBuf>,
    workspace: bool,
}

pub fn analyze_dependencies(policy: &BuildPolicy) -> Result<DependencyAnalysis, String> {
    let mut manifests = Vec::new();
    collect_manifests(&policy.root, &mut manifests).map_err(|error| error.to_string())?;
    manifests.sort();
    let mut loaded = Vec::new();
    for (kind, manifest) in manifests {
        let source = fs::read_to_string(&manifest)
            .map_err(|error| format!("could not inspect {}: {error}", manifest.display()))?;
        let value = toml::from_str::<toml::Value>(&source)
            .map_err(|error| format!("invalid {}: {error}", manifest.display()))?;
        let Some(package) = value.get("package").and_then(toml::Value::as_table) else {
            continue;
        };
        let Some(name) = package.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let directory = manifest.parent().unwrap_or(&policy.root);
        loaded.push(LoadedNode {
            public: ArchitectureNode {
                name: name.to_owned(),
                path: relative_path(&policy.root, directory),
                manifest: manifest.clone(),
            },
            kind,
            source,
            value,
        });
    }
    loaded.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.public.path.cmp(&right.public.path))
    });
    let dependencies = resolve_edges(&loaded)?;
    let mut analysis = DependencyAnalysis {
        nodes: loaded.iter().map(|node| node.public.clone()).collect(),
        dependencies,
        findings: Vec::new(),
        stats: Vec::new(),
    };
    analyze_cycles(&mut analysis, &policy.architecture);
    analyze_layers(&mut analysis, &policy.architecture);
    analyze_rules(&mut analysis, &policy.architecture);
    analysis.findings.sort_by(|left, right| {
        left.manifest
            .cmp(&right.manifest)
            .then_with(|| left.line.cmp(&right.line))
            .then_with(|| left.code.cmp(right.code))
    });
    analysis.stats = dependency_stats(&analysis);
    Ok(analysis)
}

fn collect_manifests(
    directory: &Path,
    output: &mut Vec<(ManifestKind, PathBuf)>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            let name = path.file_name().and_then(|name| name.to_str());
            if matches!(name, Some("target" | "node_modules"))
                || name.is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            collect_manifests(&path, output)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        for kind in [ManifestKind::Cargo, ManifestKind::Severian] {
            if path.file_name().and_then(|name| name.to_str()) == Some(kind.file_name()) {
                output.push((kind, path.clone()));
            }
        }
    }
    Ok(())
}

fn resolve_edges(nodes: &[LoadedNode]) -> Result<Vec<ArchitectureDependency>, String> {
    let mut by_directory = BTreeMap::new();
    let mut by_name = BTreeMap::<(ManifestKind, String), Vec<usize>>::new();
    for (index, node) in nodes.iter().enumerate() {
        let directory = node
            .public
            .manifest
            .parent()
            .unwrap_or(Path::new("."))
            .canonicalize()
            .map_err(|error| {
                format!(
                    "could not resolve package directory {}: {error}",
                    node.public.path.display()
                )
            })?;
        by_directory.insert((node.kind, directory), index);
        by_name
            .entry((node.kind, node.public.name.clone()))
            .or_default()
            .push(index);
    }
    let mut seen = BTreeSet::new();
    let mut edges = Vec::new();
    for (source_index, node) in nodes.iter().enumerate() {
        for declaration in declared_dependencies(node) {
            let target = if let Some(path) = declaration.path {
                let base = node.public.manifest.parent().unwrap_or(Path::new("."));
                let directory = base.join(path).canonicalize().map_err(|error| {
                    format!(
                        "dependency `{}` in {} has an invalid local path: {error}",
                        declaration.key,
                        node.public.manifest.display()
                    )
                })?;
                by_directory.get(&(node.kind, directory)).copied()
            } else if declaration.workspace {
                by_name
                    .get(&(node.kind, declaration.package.clone()))
                    .filter(|matches| matches.len() == 1)
                    .and_then(|matches| matches.first().copied())
            } else {
                None
            };
            let Some(target_index) = target else { continue };
            if !seen.insert((source_index, target_index)) {
                continue;
            }
            edges.push(ArchitectureDependency {
                source: source_index,
                target: target_index,
                manifest: node.public.manifest.clone(),
                line: dependency_line(&node.source, &declaration.key),
            });
        }
    }
    edges.sort_by_key(|edge| (edge.source, edge.target));
    Ok(edges)
}

fn declared_dependencies(node: &LoadedNode) -> Vec<DeclaredDependency> {
    let mut output = Vec::new();
    if let Some(table) = node
        .value
        .get("dependencies")
        .and_then(toml::Value::as_table)
    {
        extend_dependencies(table, &mut output);
    }
    if node.kind == ManifestKind::Cargo {
        if let Some(table) = node
            .value
            .get("build-dependencies")
            .and_then(toml::Value::as_table)
        {
            extend_dependencies(table, &mut output);
        }
        if let Some(targets) = node.value.get("target").and_then(toml::Value::as_table) {
            for target in targets.values().filter_map(toml::Value::as_table) {
                for name in ["dependencies", "build-dependencies"] {
                    if let Some(table) = target.get(name).and_then(toml::Value::as_table) {
                        extend_dependencies(table, &mut output);
                    }
                }
            }
        }
    }
    output
}

fn extend_dependencies(table: &toml::Table, output: &mut Vec<DeclaredDependency>) {
    for (alias, value) in table {
        let detail = value.as_table();
        output.push(DeclaredDependency {
            key: alias.clone(),
            package: detail
                .and_then(|table| table.get("package"))
                .and_then(toml::Value::as_str)
                .unwrap_or(alias)
                .to_owned(),
            path: detail
                .and_then(|table| table.get("path"))
                .and_then(toml::Value::as_str)
                .map(PathBuf::from),
            workspace: detail
                .and_then(|table| table.get("workspace"))
                .and_then(toml::Value::as_bool)
                .unwrap_or(false),
        });
    }
}

fn dependency_line(source: &str, dependency: &str) -> Option<usize> {
    source
        .lines()
        .position(|line| {
            let line = line.trim_start();
            line.strip_prefix(dependency)
                .is_some_and(|rest| rest.trim_start().starts_with('='))
        })
        .map(|index| index + 1)
}

fn analyze_cycles(analysis: &mut DependencyAnalysis, policy: &ArchitecturePolicy) {
    if !policy.deny_cycles {
        return;
    }
    let adjacency = adjacency(analysis.nodes.len(), &analysis.dependencies);
    for component in strongly_connected_components(&adjacency) {
        let self_cycle = component.len() == 1
            && adjacency[component[0]]
                .iter()
                .any(|target| *target == component[0]);
        if component.len() == 1 && !self_cycle {
            continue;
        }
        let cycle = cycle_path(&component, &adjacency);
        let edge = cycle
            .windows(2)
            .find_map(|pair| find_edge(&analysis.dependencies, pair[0], pair[1]));
        let names = cycle
            .iter()
            .map(|index| analysis.nodes[*index].name.as_str())
            .collect::<Vec<_>>()
            .join(" -> ");
        analysis.findings.push(ArchitectureFinding {
            code: "architecture::dependency_cycle",
            severity: "error",
            manifest: edge
                .map(|edge| edge.manifest.clone())
                .unwrap_or_else(|| analysis.nodes[component[0]].manifest.clone()),
            line: edge.and_then(|edge| edge.line),
            message: format!("package dependency cycle: {names}"),
        });
    }
}

fn analyze_layers(analysis: &mut DependencyAnalysis, policy: &ArchitecturePolicy) {
    if policy.layers.order.is_empty() {
        return;
    }
    let positions = policy
        .layers
        .order
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let raw_layers = analysis
        .nodes
        .iter()
        .map(|node| node_layer(node, &positions))
        .collect::<Vec<_>>();
    let inferred_roots = if policy.layers.include.is_empty() {
        let mut candidates = BTreeMap::<PathBuf, BTreeSet<usize>>::new();
        for (node, layer) in analysis.nodes.iter().zip(&raw_layers) {
            if let Some(layer) = layer {
                candidates
                    .entry(node.path.parent().unwrap_or(Path::new("")).to_path_buf())
                    .or_default()
                    .insert(*layer);
            }
        }
        let widest = candidates.values().map(BTreeSet::len).max().unwrap_or(0);
        candidates
            .into_iter()
            .filter_map(|(root, layers)| (layers.len() == widest).then_some(root))
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let in_scope = |node: &ArchitectureNode| {
        if policy.layers.include.is_empty() {
            inferred_roots.contains(node.path.parent().unwrap_or(Path::new("")))
        } else {
            policy
                .layers
                .include
                .iter()
                .any(|pattern| architecture_path_matches(pattern, &path_string(&node.path)))
        }
    };
    let layers = analysis
        .nodes
        .iter()
        .zip(raw_layers)
        .map(|(node, layer)| in_scope(node).then_some(layer).flatten())
        .collect::<Vec<_>>();
    if policy.deny_unknown_layers {
        for (index, node) in analysis.nodes.iter().enumerate() {
            if layers[index].is_none() && in_scope(node) {
                analysis.findings.push(ArchitectureFinding {
                    code: "architecture::unknown_layer",
                    severity: "error",
                    manifest: node.manifest.clone(),
                    line: None,
                    message: format!(
                        "package `{}` at `{}` is not listed in `architecture.layers.order`",
                        node.name,
                        node.path.display()
                    ),
                });
            }
        }
    }
    if !policy.deny_layer_violations {
        return;
    }
    for edge in &analysis.dependencies {
        let (Some(source), Some(target)) = (layers[edge.source], layers[edge.target]) else {
            continue;
        };
        if source < target {
            analysis.findings.push(ArchitectureFinding {
                code: "architecture::backward_layer_dependency",
                severity: "error",
                manifest: edge.manifest.clone(),
                line: edge.line,
                message: format!(
                    "layer `{}` may not depend on later layer `{}` (`{}` -> `{}`)",
                    policy.layers.order[source],
                    policy.layers.order[target],
                    analysis.nodes[edge.source].path.display(),
                    analysis.nodes[edge.target].path.display()
                ),
            });
        }
    }
}

fn analyze_rules(analysis: &mut DependencyAnalysis, policy: &ArchitecturePolicy) {
    for edge in &analysis.dependencies {
        let source = path_string(&analysis.nodes[edge.source].path);
        let target = path_string(&analysis.nodes[edge.target].path);
        let matching = policy
            .rules
            .iter()
            .filter(|rule| architecture_path_matches(&rule.from, &source))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        let denied = matching.iter().any(|rule| {
            rule.deny
                .iter()
                .any(|pattern| architecture_path_matches(pattern, &target))
        });
        let allow_lists = matching
            .iter()
            .filter(|rule| !rule.allow.is_empty())
            .collect::<Vec<_>>();
        let outside_allow = !allow_lists.is_empty()
            && !allow_lists.iter().any(|rule| {
                rule.allow
                    .iter()
                    .any(|pattern| architecture_path_matches(pattern, &target))
            });
        if denied || outside_allow {
            let reason = if denied {
                "matches an explicit deny rule"
            } else {
                "is outside the explicit allow list"
            };
            analysis.findings.push(ArchitectureFinding {
                code: "architecture::forbidden_dependency",
                severity: "error",
                manifest: edge.manifest.clone(),
                line: edge.line,
                message: format!("forbidden dependency `{source}` -> `{target}`: {reason}"),
            });
        }
    }
}

fn node_layer(node: &ArchitectureNode, positions: &HashMap<&str, usize>) -> Option<usize> {
    let directory = node.path.file_name().and_then(|name| name.to_str());
    directory
        .and_then(|name| positions.get(name).copied())
        .or_else(|| positions.get(node.name.as_str()).copied())
        .or_else(|| {
            node.name
                .strip_prefix("severian-")
                .and_then(|name| positions.get(name).copied())
        })
}

fn dependency_stats(analysis: &DependencyAnalysis) -> Vec<DependencyStat> {
    let mut fan_in = vec![0; analysis.nodes.len()];
    let mut fan_out = vec![0; analysis.nodes.len()];
    for edge in &analysis.dependencies {
        fan_out[edge.source] += 1;
        fan_in[edge.target] += 1;
    }
    let mut stats = (0..analysis.nodes.len())
        .map(|node| DependencyStat {
            node,
            fan_in: fan_in[node],
            fan_out: fan_out[node],
        })
        .collect::<Vec<_>>();
    stats.sort_by(|left, right| {
        right
            .fan_out
            .cmp(&left.fan_out)
            .then_with(|| right.fan_in.cmp(&left.fan_in))
            .then_with(|| {
                analysis.nodes[left.node]
                    .path
                    .cmp(&analysis.nodes[right.node].path)
            })
    });
    stats
}

fn find_edge(
    edges: &[ArchitectureDependency],
    source: usize,
    target: usize,
) -> Option<&ArchitectureDependency> {
    edges
        .iter()
        .find(|edge| edge.source == source && edge.target == target)
}

fn relative_path(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

fn path_string(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}
