#![forbid(unsafe_code)]

use severian_ast::{ImportDeclaration, ImportSubject, Item, Module};
use severian_diagnostics::Diagnostic;
use severian_source::SourceFile;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ResolvedModule {
    pub path: PathBuf,
    pub package: PackageId,
    pub ast: Module,
}

#[derive(Debug, Clone)]
pub struct ModuleGraph {
    /// Dependency-first initialization order; the root is always last.
    pub modules: Vec<ResolvedModule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackageId(pub u32);

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
    let mut resolver = Resolver::new(packages);
    resolver.visit(root, packages.root)?;
    Ok(ModuleGraph {
        modules: resolver.order,
    })
}

struct Resolver<'a> {
    packages: &'a PackageGraph,
    parsed: BTreeMap<PathBuf, Module>,
    visiting: Vec<PathBuf>,
    visited: BTreeSet<PathBuf>,
    order: Vec<ResolvedModule>,
}

impl<'a> Resolver<'a> {
    fn new(packages: &'a PackageGraph) -> Self {
        Self {
            packages,
            parsed: BTreeMap::new(),
            visiting: Vec::new(),
            visited: BTreeSet::new(),
            order: Vec::new(),
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

        let source = SourceFile::load(&canonical).map_err(|error| {
            Diagnostic::new(
                "E000001",
                format!("could not read {}: {error}", canonical.display()),
                None,
            )
        })?;
        let tokens = severian_lexer::scan(&source)?;
        let ast = severian_parser::parse(&tokens)?;
        self.parsed.insert(canonical.clone(), ast.clone());
        self.visiting.push(canonical.clone());
        for import in ast.items.iter().filter_map(|item| match item {
            Item::Import(import) => Some(import),
            _ => None,
        }) {
            if let Some((dependency, dependency_package)) =
                source_import(&canonical, package, import, self.packages)?
            {
                self.visit(&dependency, dependency_package)?;
            }
        }
        self.visiting.pop();
        self.visited.insert(canonical.clone());
        self.order.push(ResolvedModule {
            path: canonical,
            package,
            ast,
        });
        Ok(())
    }
}

fn source_import(
    importer: &Path,
    importer_package: PackageId,
    import: &ImportDeclaration,
    packages: &PackageGraph,
) -> Result<Option<(PathBuf, PackageId)>, Diagnostic> {
    if let Some(package) = &import.source {
        return package_source(importer_package, import, package, packages).map(Some);
    }
    let locator = match &import.subject {
        ImportSubject::Name(name) => {
            return package_source(importer_package, import, name, packages).map(Some)
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
            "import \"dependency.sev\" as dependency\n",
        )
        .unwrap();
        let graph = resolve(&root.join("root.sev")).unwrap();
        assert_eq!(graph.modules.len(), 2);
        assert!(graph.modules[0].path.ends_with("dependency.sev"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn declaration_only_cycles_do_not_create_an_initialization_cycle() {
        let root = temporary();
        std::fs::write(
            root.join("a.sev"),
            "import \"b.sev\" as b\ndef a():\n    pass\n",
        )
        .unwrap();
        std::fs::write(
            root.join("b.sev"),
            "import \"a.sev\" as a\ndef b():\n    pass\n",
        )
        .unwrap();
        let graph = resolve(&root.join("a.sev")).unwrap();
        assert_eq!(graph.modules.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cycles_with_runtime_initializers_are_rejected() {
        let root = temporary();
        std::fs::write(root.join("a.sev"), "import \"b.sev\" as b\nvalue := 1\n").unwrap();
        std::fs::write(root.join("b.sev"), "import \"a.sev\" as a\n").unwrap();
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
