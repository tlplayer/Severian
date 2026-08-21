#![forbid(unsafe_code)]

use severian_ast::{ImportDeclaration, ImportSubject, Item, Module};
use severian_diagnostics::Diagnostic;
use severian_source::SourceFile;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ResolvedModule {
    pub path: PathBuf,
    pub ast: Module,
}

#[derive(Debug, Clone)]
pub struct ModuleGraph {
    /// Dependency-first initialization order; the root is always last.
    pub modules: Vec<ResolvedModule>,
}

pub fn resolve(root: &Path) -> Result<ModuleGraph, Diagnostic> {
    let mut resolver = Resolver::default();
    resolver.visit(root)?;
    Ok(ModuleGraph {
        modules: resolver.order,
    })
}

#[derive(Default)]
struct Resolver {
    parsed: BTreeMap<PathBuf, Module>,
    visiting: Vec<PathBuf>,
    visited: BTreeSet<PathBuf>,
    order: Vec<ResolvedModule>,
}

impl Resolver {
    fn visit(&mut self, path: &Path) -> Result<(), Diagnostic> {
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
            if let Some(dependency) = source_import(&canonical, import)? {
                self.visit(&dependency)?;
            }
        }
        self.visiting.pop();
        self.visited.insert(canonical.clone());
        self.order.push(ResolvedModule {
            path: canonical,
            ast,
        });
        Ok(())
    }
}

fn source_import(
    importer: &Path,
    import: &ImportDeclaration,
) -> Result<Option<PathBuf>, Diagnostic> {
    if import.source.is_some() {
        return Ok(None);
    }
    let locator = match &import.subject {
        // Bare names belong to package/semantic resolution. Reject them until
        // that resolver can provide a concrete package identity; never guess a
        // sibling `<name>.sev` file or silently discard the import.
        ImportSubject::Name(name) => {
            return Err(Diagnostic::new(
                "E000124",
                format!("package import `{name}` has not been resolved"),
                Some(import.span),
            ))
        }
        ImportSubject::Locator(locator) if locator.contains(':') => return Ok(None),
        ImportSubject::Locator(locator) => locator.clone(),
    };
    let path = importer
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(locator);
    if path.is_file() {
        Ok(Some(path))
    } else {
        Err(Diagnostic::new(
            "E000123",
            format!("could not resolve source import `{}`", path.display()),
            Some(import.span),
        ))
    }
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
}
