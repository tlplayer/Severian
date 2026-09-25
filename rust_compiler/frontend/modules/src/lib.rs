#![forbid(unsafe_code)]

use severian_ast::{ImportDeclaration, ImportSubject, Item, Module};
use severian_diagnostics::Diagnostic;
use severian_source::{SourceFile, SourceMap};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ResolvedModule {
    /// Stable within a package graph and derived from package identity plus the
    /// module's package-relative source path, never from traversal order.
    pub id: ModuleId,
    pub path: PathBuf,
    /// The exact source used to produce `ast`, including its graph-unique ID.
    pub source: SourceFile,
    pub package: PackageId,
    pub ast: Module,
    /// Source imports resolved while constructing the graph. External XXI
    /// imports deliberately have no module edge.
    pub imports: Vec<ResolvedImport>,
}

#[derive(Debug, Clone)]
pub struct ModuleGraph {
    /// Dependency-first initialization order; the root is always last.
    pub modules: Vec<ResolvedModule>,
    pub policies: BTreeMap<PackageId, PackagePolicy>,
}

impl ModuleGraph {
    /// Reporting boundary shared by compilation and source tooling. Enrich the
    /// structured error before any caller converts it to a string.
    pub fn contextualize(&self, diagnostic: Diagnostic, stage: &str) -> Diagnostic {
        let mut diagnostic = diagnostic.with_sources(self.modules.iter().map(|module| module.source.clone()));
        if diagnostic.context.is_none() {
            let owner = diagnostic.span.and_then(|span| self.modules.iter().find(|module| module.source.id == span.source));
            if let Some(module) = owner.or_else(|| self.modules.last()) {
                let mut package = format!("package#{} ({})", module.package.0, module.path.display());
                for directory in module.path.ancestors().skip(1) {
                    let manifest = directory.join("package.json");
                    if !manifest.is_file() { continue; }
                    if let Some(value) = std::fs::read_to_string(&manifest).ok()
                        .and_then(|text| json5::from_str::<serde_json::Value>(&text).ok()) {
                        if let Some(name) = value["package"]["name"].as_str() {
                            let version = value["package"]["version"].as_str().unwrap_or("unspecified");
                            package = format!("{name}@{version} ({})", directory.display());
                        }
                    }
                    break;
                }
                diagnostic.context = Some(severian_diagnostics::DiagnosticContext::source(package, stage));
            }
        }
        let additional = std::mem::take(&mut diagnostic.additional);
        diagnostic.additional = Box::new((*additional).into_iter()
            .map(|additional| self.contextualize(additional, stage)).collect());
        diagnostic
    }
}

#[derive(Debug, Clone)]
pub struct PackagePolicy {
    pub narrow_imports: bool,
    pub library: PathBuf,
    pub explicit_imports: bool,
    pub lint_enabled: bool,
    pub explicit_imports_level: String,
}

/// Package planning groups declaration cycles; semantic analysis resolves the
/// combined declaration environment before checking bodies or producing MIR.
pub fn package_resolution_units(graph: &ModuleGraph) -> Result<Vec<Vec<PackageId>>, Diagnostic> {
    use severian_graph::{Dependency, Graph, Requirement};
    let modules: BTreeMap<_, _> = graph.modules.iter().map(|module| (module.id, module)).collect();
    let root = graph.modules.last().map(|module| module.package);
    let mut nodes: Vec<_> = graph.modules.iter().map(|module| module.package).collect::<BTreeSet<_>>().into_iter().collect();
    nodes.sort_by_key(|package| (Some(*package) == root, *package));
    let mut edges = Vec::new();
    for module in &graph.modules {
        for import in &module.imports {
            let target = modules.get(&import.module).ok_or_else(|| Diagnostic::new("E000128", "import refers to an undiscovered module", Some(import.span)))?;
            let edge = Dependency { source: module.package, target: target.package, requirement: Requirement::Declaration };
            if module.package != target.package && !edges.contains(&edge) { edges.push(edge); }
        }
    }
    let dependencies = Graph::new(nodes, edges).map_err(|message| Diagnostic::new("E000128", message, None))?;
    Ok(dependencies.condensation().0)
}

/// Compatibility view. Compilation scheduling retains SCC units below.
pub fn package_order(graph: &ModuleGraph) -> Result<Vec<PackageId>, Diagnostic> {
    Ok(package_resolution_units(graph)?.into_iter().flatten().collect())
}

pub fn order_packages(graph: &mut ModuleGraph) -> Result<(), Diagnostic> {
    let units = package_resolution_units(graph)?;
    let positions: BTreeMap<_, _> = units.into_iter().enumerate()
        .flat_map(|(unit, members)| members.into_iter().map(move |member| (member, unit))).collect();
    // Preserve discovery/initialization order inside a mutually dependent unit,
    // including the requested root. Never impose an order inside an SCC.
    graph.modules.sort_by_key(|module| positions[&module.package]);
    Ok(())
}

fn package_policies(packages: &PackageGraph) -> Result<BTreeMap<PackageId, PackagePolicy>, Diagnostic> {
    let mut policies = BTreeMap::new();
    for package in packages.packages.values() {
        let path = package.root.join("package.json");
        if !path.is_file() { continue; }
        let text = std::fs::read_to_string(&path).map_err(|e| Diagnostic::new("E000125", e.to_string(), None))?;
        let value: serde_json::Value = json5::from_str(&text).map_err(|e| Diagnostic::new("E000125", e.to_string(), None))?;
        let narrow_imports = match value.get("language").and_then(|v|v.get("explicit-imports")) {
            None => false,
            Some(serde_json::Value::Bool(enabled)) => *enabled,
            _ => return Err(Diagnostic::new("E000125", format!("{}: language.explicit-imports requires a boolean", path.display()), None)),
        };
        let explicit_imports = match value.get("lint").and_then(|v|v.get("explicit-imports")) {
            None => true,
            Some(serde_json::Value::Bool(enabled)) => *enabled,
            _ => return Err(Diagnostic::new("E000125", format!("{}: lint.explicit-imports requires a boolean", path.display()), None)),
        };
        let lint_enabled = value["lint"]["enabled"].as_bool().unwrap_or(true);
        let explicit_imports_level = value["lint"]["rules"]["L0015"].as_str().unwrap_or("warning").to_owned();
        policies.insert(package.id, PackagePolicy { narrow_imports, library: std::fs::canonicalize(&package.library).unwrap_or_else(|_|package.library.clone()), explicit_imports, lint_enabled, explicit_imports_level });
    }
    Ok(policies)
}

/// Package boundaries are enforced during compilation, after refactoring tools
/// have had an opportunity to inspect and replace legacy wildcard imports.
pub fn validate_import_policy(graph: &ModuleGraph) -> Result<(), Diagnostic> {
    validate_import_policy_impl(graph)
        .map_err(|diagnostic| graph.contextualize(diagnostic, "import policy"))
}

fn validate_import_policy_impl(graph: &ModuleGraph) -> Result<(), Diagnostic> {
    package_order(graph)?;
    for module in &graph.modules {
        let mut bindings = BTreeMap::new();
        for import in module.ast.items.iter().filter_map(|i|if let Item::Import(i)=i {Some(i)}else{None}) {
            let spelling = module.source.text.get(import.span.start as usize..import.span.end as usize).unwrap_or("");
            let namespace = spelling.trim_start().starts_with("import \"") || spelling.trim_start().starts_with("import '");
            if import.is_wildcard() && !namespace && graph.policies.get(&module.package).is_some_and(|policy| policy.narrow_imports) {
                return Err(Diagnostic::new("E000125", "wildcard imports are forbidden by language.explicit-imports", Some(import.span))
                    .with_help("enable lint.enabled for automatic import correction, or run `sev --lint`")
                    .with_source(module.source.clone()));
            }
            let binding = import.alias.as_deref().or_else(||match &import.subject {
                ImportSubject::Name(n) if n != "*"=>Some(n.as_str()),
                ImportSubject::Name(_)=>None,
                ImportSubject::Locator(_)=>import.source.as_deref(),
            });
            if let Some(name) = binding {
                if let Some(previous) = bindings.insert(name, import.span) {
                    return Err(Diagnostic::new("E000203",format!("duplicate import binding `{name}`"),Some(import.span))
                        .with_label(previous, "previous import introduces this name")
                        .with_help(format!("remove one import of `{name}` or give it a distinct `as` alias"))
                        .with_source(module.source.clone()));
                }
            }

        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackageId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedImport {
    pub span: severian_source::Span,
    pub module: ModuleId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPackage {
    pub id: PackageId,
    pub root: PathBuf,
    pub library: PathBuf,
    pub dependencies: BTreeMap<String, PackageId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageGraph {
    pub root: PackageId,
    pub packages: BTreeMap<PackageId, ResolvedPackage>,
}

pub fn resolve(root: &Path) -> Result<ModuleGraph, Diagnostic> {
    let package = PackageId(0);
    let graph = PackageGraph {
        root: package,
        packages: BTreeMap::from([(
            package,
            ResolvedPackage {
                id: package,
                root: root.parent().unwrap_or_else(|| Path::new(".")).to_owned(),
                library: root.to_owned(),
                dependencies: BTreeMap::new(),
            },
        )]),
    };
    resolve_with_packages(root, &graph)
}

/// Resolves source locators and package imports to concrete module roots.
/// Package identity comes from the caller's manifest context; the resolver
/// never guesses public package names from filenames.
pub fn resolve_with_packages(
    root: &Path,
    packages: &PackageGraph,
) -> Result<ModuleGraph, Diagnostic> {
    resolve_with_packages_and_max_errors(root, packages, 5)
}

pub fn resolve_with_packages_and_max_errors(
    root: &Path,
    packages: &PackageGraph,
    max_errors: usize,
) -> Result<ModuleGraph, Diagnostic> {
    resolve_with_packages_and_additional_roots(root, packages, &[], max_errors)
}

/// Resolves the root program plus compiler-provided registry packages whose
/// trait implementations must participate even when applications do not
/// import their ordinary symbols.
pub fn resolve_with_packages_and_additional_roots(
    root: &Path,
    packages: &PackageGraph,
    additional_roots: &[(PathBuf, PackageId)],
    max_errors: usize,
) -> Result<ModuleGraph, Diagnostic> {
    let mut resolver = Resolver::new(packages, max_errors);
    for (path, package) in additional_roots {
        resolver.visit(path, *package)?;
    }
    resolver.visit(root, packages.root)?;
    let canonical_root = std::fs::canonicalize(root).map_err(|error| {
        Diagnostic::new(
            "E000001",
            format!("could not read {}: {error}", root.display()),
            None,
        )
    })?;
    if let Some(position) = resolver
        .order
        .iter()
        .position(|module| module.path == canonical_root)
    {
        let root = resolver.order.remove(position);
        resolver.order.push(root);
    }
    Ok(ModuleGraph {
        modules: resolver.order,
        policies: package_policies(packages)?,
    })
}

struct Resolver<'a> {
    packages: &'a PackageGraph,
    parsed: BTreeMap<PathBuf, Module>,
    visiting: Vec<PathBuf>,
    visited: BTreeSet<PathBuf>,
    order: Vec<ResolvedModule>,
    module_ids: BTreeMap<PathBuf, ModuleId>,
    import_edges: BTreeMap<PathBuf, Vec<ResolvedImport>>,
    sources: SourceMap,
    max_errors: usize,
}

impl<'a> Resolver<'a> {
    fn new(packages: &'a PackageGraph, max_errors: usize) -> Self {
        Self {
            packages,
            parsed: BTreeMap::new(),
            visiting: Vec::new(),
            visited: BTreeSet::new(),
            order: Vec::new(),
            module_ids: BTreeMap::new(),
            import_edges: BTreeMap::new(),
            sources: SourceMap::new(),
            max_errors: max_errors.max(1),
        }
    }

    fn visit(&mut self, path: &Path, package: PackageId) -> Result<(), Diagnostic> {
        let canonical = std::fs::canonicalize(path).map_err(|error| {
            Diagnostic::new(
                "E000001",
                format!("could not read {}: {error}", path.display()),
                None,
            )
        })?;
        // Relative imports can enter a nested dependency package. Resolve its
        // imports against the owning manifest, not the caller's dependencies.
        let package = self
            .packages
            .packages
            .values()
            .filter(|candidate| canonical.starts_with(&candidate.root))
            .max_by_key(|candidate| candidate.root.components().count())
            .map_or(package, |candidate| candidate.id);
        if self.visited.contains(&canonical) {
            return Ok(());
        }
        if let Some(cycle_start) = self.visiting.iter().position(|path| path == &canonical) {
            let cycle = &self.visiting[cycle_start..];
            if cycle
                .iter()
                .any(|path| self.parsed.get(path).is_some_and(has_runtime_initializer))
            {
                return Err(Diagnostic::new(
                    "E000122",
                    format!(
                        "runtime module initialization cycle: {}",
                        cycle
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>()
                            .join(" -> ")
                    ),
                    None,
                ));
            }
            return Ok(());
        }

        let source_id = self.sources.load(&canonical).map_err(|error| {
            Diagnostic::new(
                "E000001",
                format!("could not read {}: {error}", canonical.display()),
                None,
            )
        })?;
        let source = self
            .sources
            .get(source_id)
            .expect("a newly loaded source is present in its source map")
            .clone();
        let tokens = severian_lexer::scan(&source)
            .map_err(|diagnostic| diagnostic.with_source(source.clone()))?;
        let ast = severian_parser::parse_with_max_errors(&tokens, self.max_errors)
            .map_err(|diagnostic| diagnostic.with_source(source.clone()))?;
        self.parsed.insert(canonical.clone(), ast.clone());
        let module_id = module_id(&canonical, package, self.packages)?;
        self.module_ids.insert(canonical.clone(), module_id);
        self.visiting.push(canonical.clone());
        for import in ast.items.iter().filter_map(|item| match item {
            Item::Import(import) => Some(import),
            _ => None,
        }) {
            if let Some((dependency, dependency_package)) =
                source_import(&canonical, package, import, self.packages)
                    .map_err(|diagnostic| diagnostic.with_source(source.clone()))?
            {
                self.visit(&dependency, dependency_package)?;
                let dependency = std::fs::canonicalize(&dependency).map_err(|error| {
                    Diagnostic::new(
                        "E000001",
                        format!("could not read {}: {error}", dependency.display()),
                        Some(import.span),
                    )
                    .with_source(source.clone())
                })?;
                let dependency_id = *self
                    .module_ids
                    .get(&dependency)
                    .expect("visited source imports have module identities");
                self.import_edges
                    .entry(canonical.clone())
                    .or_default()
                    .push(ResolvedImport {
                        span: import.span,
                        module: dependency_id,
                    });
            }
        }
        self.visiting.pop();
        self.visited.insert(canonical.clone());
        let imports = self.import_edges.remove(&canonical).unwrap_or_default();
        self.order.push(ResolvedModule {
            id: module_id,
            path: canonical.clone(),
            source: self
                .sources
                .get(source_id)
                .expect("a resolved source remains present in its source map")
                .clone(),
            package,
            ast,
            imports,
        });
        Ok(())
    }
}

fn module_id(
    path: &Path,
    package: PackageId,
    packages: &PackageGraph,
) -> Result<ModuleId, Diagnostic> {
    let root = packages
        .packages
        .get(&package)
        .map(|package| package.root.as_path())
        .ok_or_else(|| {
            Diagnostic::new(
                "E000125",
                format!("module belongs to unknown package {package:?}"),
                None,
            )
        })?;
    let relative = path.strip_prefix(root).unwrap_or(path);
    // Graph indices vary between independent package builds and consumers.
    // Bind module identity to its declared package and package-relative path.
    let identity = std::fs::read_to_string(root.join("package.json")).ok()
        .and_then(|text| json5::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| Some(format!("{}@{}", value["package"]["name"].as_str()?, value["package"]["version"].as_str()?)))
        .unwrap_or_else(|| root.to_string_lossy().into_owned());
    let key = format!("{identity}:{}", relative.to_string_lossy().replace('\\', "/"));
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut hash = OFFSET;
    for byte in key.as_bytes() {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    Ok(ModuleId(hash))
}

fn source_import(
    importer: &Path,
    importer_package: PackageId,
    import: &ImportDeclaration,
    packages: &PackageGraph,
) -> Result<Option<(PathBuf, PackageId)>, Diagnostic> {
    if matches!(import.subject, ImportSubject::Name(_)) && import.source.as_deref() == Some("xxi") {
        return Ok(None);
    }
    if let (ImportSubject::Name(_), Some(package)) = (&import.subject, &import.source) {
        return package_source(importer_package, import, package, packages).map(Some);
    }
    let locator = match &import.subject {
        ImportSubject::Name(name) => {
            return package_source(importer_package, import, name, packages).map(Some)
        }
        ImportSubject::Locator(locator) if locator.starts_with("package:") => {
            let locator = &locator["package:".len()..];
            let (alias, member) = locator.split_once('/').unwrap_or((locator, ""));
            let (library, dependency) =
                package_source(importer_package, import, alias, packages)?;
            if member.is_empty() {
                return Ok(Some((library, dependency)));
            }
            let root = &packages.packages[&dependency].root;
            let path = std::fs::canonicalize(root.join(member)).map_err(|error| {
                Diagnostic::new(
                    "E000123",
                    format!("could not resolve package source import `{locator}`: {error}"),
                    Some(import.span),
                )
            })?;
            let root = std::fs::canonicalize(root).map_err(|error| {
                Diagnostic::new("E000123", error.to_string(), Some(import.span))
            })?;
            if !path.starts_with(root) || !path.is_file() {
                return Err(Diagnostic::new(
                    "E000123",
                    format!("package source import `{locator}` must name a file inside its package"),
                    Some(import.span),
                ));
            }
            return Ok(Some((path, dependency)));
        }
        ImportSubject::Locator(locator) if locator.contains(':') => return Ok(None),
        ImportSubject::Locator(locator) => locator.clone(),
    };
    let path = importer
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(locator);
    if path.is_file() {
        Ok(Some((path, importer_package)))
    } else {
        Err(Diagnostic::new(
            "E000123",
            format!("could not resolve source import `{}`", path.display()),
            Some(import.span),
        ))
    }
}

fn package_source(
    importer_package: PackageId,
    import: &ImportDeclaration,
    package: &str,
    packages: &PackageGraph,
) -> Result<(PathBuf, PackageId), Diagnostic> {
    let current = packages.packages.get(&importer_package).ok_or_else(|| {
        Diagnostic::new(
            "E000125",
            format!("module belongs to unknown package {:?}", importer_package),
            Some(import.span),
        )
    })?;
    let dependency = current.dependencies.get(package).ok_or_else(|| {
        Diagnostic::new(
            "E000124",
            format!("package import `{package}` has not been resolved"),
            Some(import.span),
        )
    })?;
    let dependency = packages.packages.get(dependency).ok_or_else(|| {
        Diagnostic::new(
            "E000125",
            format!("package import `{package}` resolves to a missing package node"),
            Some(import.span),
        )
    })?;
    Ok((dependency.library.clone(), dependency.id))
}

fn has_runtime_initializer(module: &Module) -> bool {
    module
        .items
        .iter()
        .any(|item| matches!(item, Item::Binding(_) | Item::Expression(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn temporary() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "severian-modules-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn dependencies_are_ordered_before_the_root() {
        let root = temporary();
        std::fs::write(root.join("dependency.sev"), "value := 1\n").unwrap();
        std::fs::write(
            root.join("root.sev"),
            "import * from \"dependency.sev\" as dependency\n",
        )
        .unwrap();
        let graph = resolve(&root.join("root.sev")).unwrap();
        assert_eq!(graph.modules.len(), 2);
        assert!(graph.modules[0].path.ends_with("dependency.sev"));
        assert_ne!(graph.modules[0].source.id, graph.modules[1].source.id);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disconnected_registry_packages_do_not_replace_the_requested_root() {
        let directory = temporary();
        let root = directory.join("app/main.sev");
        let registry = directory.join("registry/lib.sev");
        for file in [&root, &registry] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, "def example():\n    pass\n").unwrap();
        }
        let packages = PackageGraph {
            root: PackageId(0),
            packages: BTreeMap::from([
                (PackageId(0), ResolvedPackage {
                    id: PackageId(0), root: root.parent().unwrap().to_owned(),
                    library: root.clone(), dependencies: BTreeMap::new(),
                }),
                (PackageId(9), ResolvedPackage {
                    id: PackageId(9), root: registry.parent().unwrap().to_owned(),
                    library: registry.clone(), dependencies: BTreeMap::new(),
                }),
            ]),
        };
        let mut graph = resolve_with_packages_and_additional_roots(
            &root, &packages, &[(registry, PackageId(9))], 5,
        ).unwrap();
        order_packages(&mut graph).unwrap();
        assert_eq!(graph.modules.last().unwrap().package, PackageId(0));
        assert_eq!(graph.modules.last().unwrap().path, std::fs::canonicalize(root).unwrap());
        assert_eq!(graph.modules[0].package, PackageId(9));
        // Repeated scheduling must preserve both dependency and root order.
        let paths: Vec<_> = graph.modules.iter().map(|module| module.path.clone()).collect();
        order_packages(&mut graph).unwrap();
        assert_eq!(paths, graph.modules.iter().map(|module| module.path.clone()).collect::<Vec<_>>());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn declaration_only_cycles_do_not_create_an_initialization_cycle() {
        let root = temporary();
        std::fs::write(
            root.join("a.sev"),
            "import * from \"b.sev\" as b\ndef a():\n    pass\n",
        )
        .unwrap();
        std::fs::write(
            root.join("b.sev"),
            "import * from \"a.sev\" as a\ndef b():\n    pass\n",
        )
        .unwrap();
        let graph = resolve(&root.join("a.sev")).unwrap();
        assert_eq!(graph.modules.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cycles_with_runtime_initializers_are_rejected() {
        let root = temporary();
        std::fs::write(root.join("a.sev"), "import * from \"b.sev\" as b\nvalue := 1\n").unwrap();
        std::fs::write(root.join("b.sev"), "import * from \"a.sev\" as a\n").unwrap();
        let error = resolve(&root.join("a.sev")).unwrap_err();
        assert_eq!(error.code, "E000122");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bare_package_names_are_not_guessed_as_sibling_sources() {
        let root = temporary();
        std::fs::write(root.join("io.sev"), "value := 1\n").unwrap();
        std::fs::write(root.join("root.sev"), "import io\n").unwrap();
        let error = resolve(&root.join("root.sev")).unwrap_err();
        assert_eq!(error.code, "E000124");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn xxi_interface_imports_are_not_package_dependencies() {
        let root = temporary();
        std::fs::write(root.join("root.sev"), "import c from xxi\n").unwrap();
        let graph = resolve(&root.join("root.sev")).unwrap();
        assert_eq!(graph.modules.len(), 1);
        assert!(graph.modules[0].imports.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_imports_use_only_the_supplied_manifest_context() {
        let root = temporary();
        let package = root.join("tensor/lib.sev");
        std::fs::create_dir_all(package.parent().unwrap()).unwrap();
        std::fs::write(&package, "def shape():\n    pass\n").unwrap();
        std::fs::write(root.join("root.sev"), "import tensor\n").unwrap();
        let root_package = PackageId(0);
        let tensor_package = PackageId(1);
        let packages = PackageGraph {
            root: root_package,
            packages: BTreeMap::from([
                (
                    root_package,
                    ResolvedPackage {
                        id: root_package,
                        root: root.clone(),
                        library: root.join("root.sev"),
                        dependencies: BTreeMap::from([("tensor".into(), tensor_package)]),
                    },
                ),
                (
                    tensor_package,
                    ResolvedPackage {
                        id: tensor_package,
                        root: package.parent().unwrap().to_owned(),
                        library: package.clone(),
                        dependencies: BTreeMap::new(),
                    },
                ),
            ]),
        };
        let graph = resolve_with_packages(&root.join("root.sev"), &packages).unwrap();
        assert_eq!(graph.modules.len(), 2);
        assert_eq!(
            graph.modules[0].path,
            std::fs::canonicalize(package).unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selective_imports_resolve_their_declared_source_package() {
        let root = temporary();
        let package = root.join("io/lib.sev");
        std::fs::create_dir_all(package.parent().unwrap()).unwrap();
        std::fs::write(&package, "def print(value: string):\n    pass\n").unwrap();
        std::fs::write(root.join("root.sev"), "import print from io\n").unwrap();
        let root_package = PackageId(0);
        let io_package = PackageId(1);
        let packages = PackageGraph {
            root: root_package,
            packages: BTreeMap::from([
                (
                    root_package,
                    ResolvedPackage {
                        id: root_package,
                        root: root.clone(),
                        library: root.join("root.sev"),
                        dependencies: BTreeMap::from([("io".into(), io_package)]),
                    },
                ),
                (
                    io_package,
                    ResolvedPackage {
                        id: io_package,
                        root: package.parent().unwrap().to_owned(),
                        library: package.clone(),
                        dependencies: BTreeMap::new(),
                    },
                ),
            ]),
        };
        let graph = resolve_with_packages(&root.join("root.sev"), &packages).unwrap();
        assert_eq!(graph.modules.len(), 2);
        assert_eq!(
            graph.modules[0].path,
            std::fs::canonicalize(package).unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
