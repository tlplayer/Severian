//! Demand-driven import binding. A wildcard is an edge to search, not an
//! instruction to copy the dependency's export table into every consumer.
use super::{insert_binding, ExportMap, ProgramIndex, Resolution};
use severian_ast::{ImportSubject, Item};
use severian_modules::{ModuleGraph, ModuleId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod uses;
pub use uses::collect_import_requirements;

pub type ImportRequirements = BTreeMap<ModuleId, BTreeSet<String>>;

#[derive(Debug, Clone, Default)]
pub struct ImportPlan {
    pub namespace_uses: ImportRequirements,
    /// Selected names, keyed by consumer and original import span. Source
    /// tools can render this plan; compilation consumes its resolved bindings.
    pub selections: BTreeMap<(ModuleId, u32), BTreeSet<String>>,
    pub requirements: ImportRequirements,
    pub requested_names: usize,
    pub imported_bindings: usize,
    pub export_entries: usize,
    bindings: BTreeMap<(ModuleId, String), Resolution>,
    private_namespaces: BTreeSet<(ModuleId, String)>,
}

#[derive(Clone)]
enum EdgeKind {
    Wildcard,
    Named { local: String, selected: String },
    Namespace { local: String, exported: bool },
}
#[derive(Clone)]
struct Edge {
    target: ModuleId,
    span: u32,
    kind: EdgeKind,
}
type Key = (ModuleId, String);
#[derive(Default)]
struct Request {
    base: Option<Resolution>,
    dependencies: Vec<(Key, u32, String)>,
    result: Option<Resolution>,
}

/// The work list contains individual name lookups, including failed lookups.
/// Recording requests before traversing dependencies makes declaration cycles
/// finite without caching a premature "missing" result from a recursive DFS.
pub fn resolve_required_imports(
    graph: &ModuleGraph,
    index: &ProgramIndex,
    mut requirements: ImportRequirements,
) -> ImportPlan {
    // Enum variants are available through their enum's imported declaration,
    // even when source spells only the variant (or namespace.Variant). They
    // are not separate entries in the export index. Retain the owner rather
    // than silently dropping the declaration that supplies the value.
    let mut variant_owners: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for module in &graph.modules {
        for item in &module.ast.items {
            if let Item::Enum(enumeration) = item {
                for variant in &enumeration.variants {
                    variant_owners.entry(&variant.name).or_default().insert(&enumeration.name);
                }
            }
        }
    }
    for names in requirements.values_mut() {
        let mut owners = BTreeSet::new();
        for name in names.iter() {
            let (prefix, variant) = name.rsplit_once('.').map_or(("", name.as_str()), |(prefix, variant)| (prefix, variant));
            for owner in variant_owners.get(variant).into_iter().flatten() {
                if prefix.is_empty() {
                    owners.insert((*owner).to_owned());
                } else if prefix.rsplit('.').next() != Some(owner) {
                    owners.insert(format!("{prefix}.{owner}"));
                }
            }
        }
        names.extend(owners);
    }
    let namespace_uses = requirements.clone();
    let mut edges: BTreeMap<ModuleId, Vec<Edge>> = BTreeMap::new();
    let mut private_namespaces = BTreeSet::new();
    for module in &graph.modules {
        let mut imports = Vec::new();
        for import in module.ast.items.iter().filter_map(|item| match item {
            Item::Import(i) => Some(i),
            _ => None,
        }) {
            let Some(edge) = module.imports.iter().find(|edge| edge.span == import.span) else {
                continue;
            };
            let kind = if import.is_wildcard() && import.alias.is_none() {
                EdgeKind::Wildcard
            } else if let Some(selected) = import.selected_name() {
                let local = import.alias.as_deref().unwrap_or(selected).to_owned();
                requirements
                    .entry(module.id)
                    .or_default()
                    .insert(local.clone());
                EdgeKind::Named {
                    local,
                    selected: selected.to_owned(),
                }
            } else {
                let local = import
                    .alias
                    .clone()
                    .unwrap_or_else(|| match &import.subject {
                        ImportSubject::Name(name) => name.clone(),
                        ImportSubject::Locator(path) => std::path::Path::new(path)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(path)
                            .to_owned(),
                    });
                let exported = import.alias.is_some();
                if !exported {
                    private_namespaces.insert((module.id, local.clone()));
                }
                requirements
                    .entry(module.id)
                    .or_default()
                    .insert(local.clone());
                EdgeKind::Namespace { local, exported }
            };
            imports.push(Edge {
                target: edge.module,
                span: import.span.start,
                kind,
            });
        }
        edges.insert(module.id, imports);
    }
    // An explicit opt-out preserves broad import visibility for that package.
    // This is a name inventory, not a propagated export map.
    let all_names: BTreeSet<_> = index
        .exports
        .values()
        .flat_map(|e| e.keys().cloned())
        .chain(edges.values().flatten().filter_map(|e| match &e.kind {
            EdgeKind::Named { local, .. } | EdgeKind::Namespace { local, .. } => {
                Some(local.clone())
            }
            _ => None,
        }))
        .collect();
    let mut pending = VecDeque::new();
    for (module, names) in &requirements {
        for name in names {
            enqueue(&mut pending, *module, name);
        }
    }
    let mut requests: BTreeMap<Key, Request> = BTreeMap::new();
    while let Some(key) = pending.pop_front() {
        if requests.contains_key(&key) {
            continue;
        }
        let (module, path) = &key;
        let (head, tail) = path
            .split_once('.')
            .map_or((path.as_str(), None), |(h, t)| (h, Some(t)));
        let mut request = Request {
            base: index.exports[module].get(path).cloned(),
            ..Request::default()
        };
        if !all_names.contains(head) {
            requests.insert(key, request);
            continue;
        }
        for edge in &edges[module] {
            let target_name = match &edge.kind {
                EdgeKind::Wildcard if !head.starts_with('_') => Some(path.clone()),
                EdgeKind::Named { local, selected } if local == head => Some(match tail {
                    Some(tail) => format!("{selected}.{tail}"),
                    None => selected.clone(),
                }),
                EdgeKind::Namespace { local, exported } if local == head => {
                    // Bare package namespaces are local, while aliases retain
                    // their existing public namespace contract.
                    if tail.is_none() {
                        if *exported
                            || requirements
                                .get(module)
                                .is_some_and(|names| names.contains(head))
                        {
                            request.base = Some(Resolution::Module(edge.target));
                            // Packages can expose a same-named default callable.
                            enqueue(&mut pending, edge.target, local);
                        }
                        None
                    } else {
                        Some(tail.unwrap().to_owned())
                    }
                }
                _ => None,
            };
            if let Some(name) = target_name {
                if name.split('.').any(|part| part.starts_with("__")) { continue; }
                if private_namespaces.contains(&(
                    edge.target,
                    name.split('.').next().unwrap_or(&name).to_owned(),
                )) {
                    continue;
                }
                enqueue(&mut pending, edge.target, &name);
                request
                    .dependencies
                    .push(((edge.target, name.clone()), edge.span, name));
            }
        }
        requests.insert(key, request);
    }
    // HIR owns the name-request graph and the join of semantic resolutions.
    // Package scheduling uses the same engine with package IDs and no symbols.
    let dependencies = severian_graph::Graph::new(
        requests.keys().cloned().collect(),
        requests.iter().flat_map(|(key, request)| request.dependencies.iter().map(move |(target, _, _)| {
            severian_graph::Dependency {
                source: key.clone(), target: target.clone(),
                requirement: severian_graph::Requirement::Symbol,
            }
        })).collect(),
    ).expect("discovery registers every name request before resolution");
    severian_graph::resolve_graph(&dependencies, |key| {
        let request = &requests[key];
        let mut merged = ExportMap::new();
        if let Some(base) = &request.base {
            merged.insert(key.1.clone(), base.clone());
        }
        for (dependency, import_span, _) in &request.dependencies {
            if let Some(value) = &requests[dependency].result {
                let wildcard = edges.get(&key.0).into_iter().flatten().any(|edge|
                    edge.target == dependency.0 && matches!(&edge.kind, EdgeKind::Wildcard)
                    && edge.span == *import_span);
                if super::resolution_definitions(value).iter().any(|id| {
                    let visibility = index.definitions[id].visibility;
                    visibility == super::Visibility::File || (wildcard && visibility != super::Visibility::Public)
                }) { continue; }
                insert_binding(
                    &mut merged,
                    key.1.clone(),
                    value.clone(),
                    &index.definitions,
                );
            }
        }
        let result = merged.remove(&key.1);
        if result == request.result { return false; }
        requests.get_mut(key).unwrap().result = result;
        true
    });
    let mut plan = ImportPlan {
        namespace_uses,
        requirements,
        private_namespaces,
        requested_names: requests.len(),
        ..ImportPlan::default()
    };
    for ((module, name), request) in &requests {
        if name.contains('.') {
            continue;
        }
        let Some(result) = &request.result else {
            continue;
        };
        if index.modules[module].scope.bindings.get(name) != Some(result) {
            plan.imported_bindings += 1;
        }
        plan.bindings
            .insert((*module, name.clone()), result.clone());
        for (dependency, span, selected) in &request.dependencies {
            if requests[dependency].result.is_some() {
                plan.selections
                    .entry((*module, *span))
                    .or_default()
                    .insert(selected.clone());
            }
        }
    }
    plan.export_entries = index.exports.values().map(|e| e.len()).sum::<usize>()
        + plan
            .bindings
            .keys()
            .filter(|(module, name)| {
                !index.exports[module].contains_key(name)
                    && !plan.private_namespaces.contains(&(*module, name.clone()))
            })
            .count();
    plan
}

fn enqueue(queue: &mut VecDeque<Key>, module: ModuleId, path: &str) {
    // Namespace prefixes must remain bound as well as the terminal member.
    for (offset, _) in path.match_indices('.') {
        queue.push_back((module, path[..offset].to_owned()));
    }
    queue.push_back((module, path.to_owned()));
}

pub fn apply_import_plan(index: &mut ProgramIndex, plan: &ImportPlan) {
    index.import_requirements = Some(plan.namespace_uses.clone());
    for ((module, name), resolution) in &plan.bindings {
        index
            .modules
            .get_mut(module)
            .unwrap()
            .scope
            .bindings
            .insert(name.clone(), resolution.clone());
        if !plan.private_namespaces.contains(&(*module, name.clone())) {
            index
                .exports
                .get_mut(module)
                .unwrap()
                .insert(name.clone(), resolution.clone());
        }
    }
}

#[cfg(test)]
mod tests;
