use crate::model::{Confidence, Evidence, Finding, Severity, SourceSpan};
use crate::source::{call_names, extract_functions, FunctionBody, SourceUnit};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub fn analyze(units: &[SourceUnit]) -> Vec<Finding> {
    let functions = units.iter().flat_map(extract_functions).collect::<Vec<_>>();
    let mut findings = reachability(units, &functions);
    findings.extend(cohesion(units));
    findings
}

fn reachability(units: &[SourceUnit], functions: &[FunctionBody]) -> Vec<Finding> {
    let mut by_name = BTreeMap::<&str, Vec<usize>>::new();
    for (index, function) in functions.iter().enumerate() {
        by_name.entry(&function.name).or_default().push(index);
    }
    let mut edges = vec![BTreeSet::new(); functions.len()];
    for (index, function) in functions.iter().enumerate() {
        for callee in call_names(&function.source) {
            if let Some(targets) = by_name.get(callee.as_str()) {
                edges[index].extend(targets.iter().copied());
            }
        }
    }
    let mut reachable = BTreeSet::new();
    let mut queue = functions
        .iter()
        .enumerate()
        .filter(|(_, function)| is_root(function))
        .map(|(index, _)| index)
        .collect::<VecDeque<_>>();
    while let Some(index) = queue.pop_front() {
        if !reachable.insert(index) {
            continue;
        }
        queue.extend(edges[index].iter().copied());
    }
    let mut occurrences = BTreeMap::<&str, usize>::new();
    for unit in units {
        for token in unit
            .text
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .filter(|token| !token.is_empty())
        {
            *occurrences.entry(token).or_default() += 1;
        }
    }
    functions
        .iter()
        .enumerate()
        .filter(|(index, function)| {
            !reachable.contains(index)
                && !is_root(function)
                && by_name
                    .get(function.name.as_str())
                    .is_some_and(|definitions| definitions.len() == 1)
                && occurrences.get(function.name.as_str()).copied() == Some(1)
        })
        .map(|(_, function)| {
            Finding::new(
                "unreachable_private_candidate",
                Severity::Warning,
                Confidence::High,
                SourceSpan {
                    path: function.path.clone(),
                    line: function.line,
                    column: 1,
                },
                Evidence {
                    summary: format!(
                        "private function `{}` has no supported textual call-graph edge",
                        function.name
                    ),
                    details: vec![
                        "Public APIs, mains, tests, extern functions, and registry-like functions were roots."
                            .into(),
                        "This source graph does not claim proven dead code; rustc HIR integration can promote it."
                            .into(),
                    ],
                    metrics: BTreeMap::new(),
                },
                &function.name,
            )
            .with_remediation(
                "Prove reachability or remove it",
                "Register the dynamic root explicitly, add a supported caller, or delete the function.",
            )
        })
        .collect()
}

fn is_root(function: &FunctionBody) -> bool {
    let declaration = function
        .source
        .lines()
        .next()
        .unwrap_or_default()
        .trim_start();
    function.name == "main"
        || declaration.starts_with("pub ")
        || declaration.contains("extern \"C\"")
        || function
            .path
            .components()
            .any(|component| component.as_os_str() == "tests")
        || function
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "tests.rs")
        || function.name.starts_with("register_")
}

fn cohesion(units: &[SourceUnit]) -> Vec<Finding> {
    let mut findings = Vec::new();
    for unit in units {
        let functions = extract_functions(unit);
        if functions.len() < 8 {
            continue;
        }
        let names = functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.name.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let mut parent = (0..functions.len()).collect::<Vec<_>>();
        let mut internal_edges = 0_usize;
        for (index, function) in functions.iter().enumerate() {
            for callee in call_names(&function.source) {
                if let Some(target) = names.get(callee.as_str()).copied() {
                    union(&mut parent, index, target);
                    internal_edges += 1;
                }
            }
        }
        let mut clusters = BTreeMap::<usize, Vec<&FunctionBody>>::new();
        for (index, function) in functions.iter().enumerate() {
            let root = find(&mut parent, index);
            clusters.entry(root).or_default().push(function);
        }
        let mut significant = clusters
            .into_values()
            .filter(|cluster| cluster.len() >= 3)
            .collect::<Vec<_>>();
        significant.sort_by_key(|cluster| std::cmp::Reverse(cluster.len()));
        let covered = significant.iter().map(Vec::len).sum::<usize>();
        if significant.len() < 2 || covered * 10 < functions.len() * 6 {
            continue;
        }
        let details = significant
            .iter()
            .take(4)
            .enumerate()
            .map(|(index, cluster)| {
                format!(
                    "cluster {} ({} symbols): {}",
                    index + 1,
                    cluster.len(),
                    cluster
                        .iter()
                        .take(8)
                        .map(|function| function.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>();
        let mut metrics = BTreeMap::new();
        metrics.insert("functions".into(), functions.len() as f64);
        metrics.insert("internal_call_edges".into(), internal_edges as f64);
        metrics.insert("significant_clusters".into(), significant.len() as f64);
        metrics.insert(
            "clustered_fraction".into(),
            covered as f64 / functions.len() as f64,
        );
        findings.push(
            Finding::new(
                "low_module_cohesion",
                Severity::Warning,
                Confidence::Heuristic,
                SourceSpan::file(unit.path.clone()),
                Evidence {
                    summary: "file contains multiple weakly connected function clusters".into(),
                    details,
                    metrics,
                },
                "disconnected-function-clusters",
            )
            .with_remediation(
                "Review a module split",
                "Use the clusters as evidence; keep the file intact if shared invariants dominate.",
            ),
        );
    }
    findings
}

fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        parent[right] = left;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn a_public_function_is_a_reachability_root() {
        let function = FunctionBody {
            path: PathBuf::from("src/lib.rs"),
            line: 1,
            name: "api".into(),
            source: "pub fn api() {}\n".into(),
        };
        assert!(is_root(&function));
    }
}
