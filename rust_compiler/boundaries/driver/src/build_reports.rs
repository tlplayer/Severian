//! Disposable reports shared by an update and its child compiler invocations.
use severian_driver::config::Manifest;
use severian_diagnostics::{Diagnostic, DiagnosticContext};
use severian_source::SourceFile;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const ENV: &str = "SEVERIAN_BUILD_REPORT_DIR";
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Reports {
    pub(crate) root: PathBuf,
}

impl Reports {
    pub(crate) fn begin() -> Result<Self, String> {
        let inherited = std::env::var_os(ENV).filter(|value| !value.is_empty());
        let root = match &inherited {
            Some(root) => PathBuf::from(root),
            None => std::env::current_dir().map_err(|error| error.to_string())?.join("package.pkg/debug/build"),
        };
        Self::open(root, inherited.is_none())
    }

    fn open(root: PathBuf, reset: bool) -> Result<Self, String> {
        if reset && root.exists() {
            fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
        }
        fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        Ok(Self { root })
    }

    pub(crate) fn record(&self, step: &str, category: &str, source: &Path, message: &str) -> Result<(), String> {
        if !matches!(category, "error" | "lint" | "warning") {
            return Err(format!("invalid build report category: {category}"));
        }
        let step = if step.trim().is_empty() { "build" } else { step };
        let step: String = step.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
        let directory = self.root.join(step);
        for category in ["error", "lint", "warning"] {
            fs::create_dir_all(directory.join(category)).map_err(|error| error.to_string())?;
        }
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let record = serde_json::json!({"source": source, "message": message});
        fs::write(directory.join(category).join(format!("{}-{id}.json", std::process::id())),
            serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())
    }

    fn diagnostic(&self, mut diagnostic: Diagnostic, source: &SourceFile, step: &str) -> Result<(), String> {
        // Parser recovery returns additional diagnostics. Give every one its
        // own report and source snapshot instead of hiding it in the first error.
        let additional = std::mem::take(&mut *diagnostic.additional);
        diagnostic.context = Some(DiagnosticContext::source(source.path.display().to_string(), step));
        if diagnostic.help.is_none() {
            diagnostic.help = Some(format!("correct this {step} error in {} and rebuild", source.path.display()));
        }
        diagnostic = diagnostic.with_source(source.clone());
        self.record(step, "error", &source.path, &diagnostic.to_string())?;
        for additional in additional {
            self.diagnostic(additional, source, step)?;
        }
        Ok(())
    }

    pub(crate) fn scan(&self, manifest: Option<&Manifest>, input: &Path) -> Result<(Vec<PathBuf>, bool), String> {
        let mut files = BTreeSet::new();
        if let Some(manifest) = manifest {
            for package in manifest.package_graph.packages.values() {
                collect(&package.root, &mut files)?;
            }
        } else {
            collect(input, &mut files)?;
        }
        let mut valid = Vec::new();
        let mut failed = false;
        for path in files {
            let source = match SourceFile::load(&path) {
                Ok(source) => source,
                Err(error) => {
                    self.record("source", "error", &path, &error.to_string())?;
                    failed = true;
                    continue;
                }
            };
            let tokens = match severian_lexer::scan(&source) {
                Ok(tokens) => tokens,
                Err(error) => {
                    self.diagnostic(error, &source, "lexer")?;
                    failed = true;
                    continue;
                }
            };
            match severian_parser::parse_with_max_errors(&tokens, usize::MAX) {
                Ok(_) => valid.push(path),
                Err(error) => {
                    self.diagnostic(error, &source, "parser")?;
                    failed = true;
                }
            }
        }
        Ok((valid, failed))
    }

    pub(crate) fn failure(&self) -> String {
        format!("build failed; collected reports: {}", self.root.display())
    }

    pub(crate) fn compile_error(&self, source: &Path, error: &severian_driver::CompileError) -> Result<(), String> {
        use severian_driver::CompileError;
        let step = match error {
            CompileError::Diagnostic(diagnostic) => diagnostic.context.as_ref()
                .map(|context| context.stage.as_str()).unwrap_or("semantic"),
            CompileError::Bootstrap(_) => "bootstrap",
            CompileError::Compile(_) => "compile",
            CompileError::MirVerify(_) | CompileError::MirPass(_) => "mir",
            CompileError::Lowering(_) => "lowering",
            CompileError::Mlir(_) => "mlir",
            CompileError::Backend(_) => "backend",
            CompileError::Component(_) => "component",
            CompileError::AgentIr(_) => "agent-ir",
            CompileError::NativeLink(_) => "link",
        };
        self.record(step, "error", source, &error.to_string())
    }
}

fn collect(path: &Path, files: &mut BTreeSet<PathBuf>) -> Result<(), String> {
    if path.is_file() {
        if path.extension().is_some_and(|extension| extension == "sev") {
            files.insert(path.canonicalize().map_err(|error| error.to_string())?);
        }
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|error| format!("{}: {error}", path.display()))? {
        let entry = entry.map_err(|error| error.to_string())?;
        let name = entry.file_name();
        // Fixtures and examples can intentionally be invalid. Build inventory
        // contains production source, not stale artifacts or regression inputs.
        if matches!(name.to_str(), Some("package.pkg" | ".git" | "target" | "tests" | "fixtures" | "docs")) {
            continue;
        }
        if entry.file_type().map_err(|error| error.to_string())?.is_symlink() {
            continue;
        }
        collect(&entry.path(), files)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_parse_failures_survive_until_the_next_run() {
        let root = std::env::temp_dir().join(format!("sev-build-reports-{}", std::process::id()));
        fs::create_dir_all(root.join("source")).unwrap();
        fs::write(root.join("source/first.sev"), "def first(:\n").unwrap();
        fs::write(root.join("source/second.sev"), "def second(:\n").unwrap();
        fs::write(root.join("source/valid.sev"), "value = 1\n").unwrap();
        let reports = Reports::open(root.join("reports"), true).unwrap();
        let (valid, failed) = reports.scan(None, &root.join("source")).unwrap();
        assert!(failed);
        assert_eq!(valid.len(), 1);
        assert!(fs::read_dir(reports.root.join("parser/error")).unwrap().count() >= 2);
        let child = Reports::open(reports.root.clone(), false).unwrap();
        assert!(child.root.join("parser/error").exists());
        Reports::open(reports.root.clone(), true).unwrap();
        assert!(!reports.root.join("parser/error").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
