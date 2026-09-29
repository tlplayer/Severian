//! Disposable reports shared by an update and its child compiler invocations.
use severian_driver::config::Manifest;
use severian_diagnostics::{Diagnostic, DiagnosticContext};
use severian_source::SourceFile;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::hash::{Hash, Hasher};

pub(crate) const ENV: &str = "SEVERIAN_BUILD_REPORT_DIR";

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
        let record = serde_json::json!({"source": source, "message": message,
            "status": if message.starts_with("blocked:") { "blocked" } else { "reported" },
            "callers": [source]});
        self.store(&directory.join(category), serde_json::json!([source, message]), record)
    }

    fn store(&self, directory: &Path, key: serde_json::Value, mut record: serde_json::Value) -> Result<(), String> {
        fs::create_dir_all(directory).map_err(|error| error.to_string())?;
        let lock = fs::OpenOptions::new().create(true).truncate(false).read(true).write(true)
            .open(self.root.join(".reports.lock")).map_err(|error| error.to_string())?;
        lock.lock().map_err(|error| error.to_string())?;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        key.to_string().hash(&mut hash);
        let mut suffix = 0usize;
        let destination = loop {
            let destination = directory.join(format!("{:016x}-{suffix}.json", hash.finish()));
            if destination.is_file() {
                let mut existing: serde_json::Value = serde_json::from_slice(
                    &fs::read(&destination).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
                if existing["identity"] != key { suffix += 1; continue; }
                if let Some(callers) = record["callers"].as_array() {
                    for caller in callers {
                        let known = existing["callers"].as_array_mut().ok_or("invalid report callers")?;
                        if !known.contains(caller) { known.push(caller.clone()); }
                    }
                }
                record = existing;
            }
            break destination;
        };
        record["identity"] = key;
        fs::write(destination, serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())
    }

    fn diagnostic(&self, diagnostic: Diagnostic, source: &SourceFile, step: &str) -> Result<(), String> {
        self.report_diagnostic(diagnostic.with_source(source.clone()), &source.path, step)
    }

    fn report_diagnostic(&self, mut diagnostic: Diagnostic, caller: &Path, fallback: &str) -> Result<(), String> {
        let additional = std::mem::take(&mut *diagnostic.additional);
        let step = diagnostic.context.as_ref().map(|context| context.stage.clone())
            .unwrap_or_else(|| fallback.to_owned());
        let location = diagnostic.span.and_then(|span| diagnostic.sources.iter()
            .find(|source| source.id == span.source).map(|source| source.path.clone()))
            .unwrap_or_else(|| caller.to_owned());
        if diagnostic.context.is_none() {
            let owner = owner_manifest(&location).unwrap_or_else(|| location.clone());
            diagnostic.context = Some(DiagnosticContext::source(owner.display().to_string(), &step));
        }
        if diagnostic.help.is_none() {
            diagnostic.help = Some(format!("correct the highlighted {} error in {}", step, location.display()));
        }
        let span = diagnostic.span.map(|span| (span.start, span.end));
        let key = serde_json::json!([diagnostic.code, location, span, diagnostic.message]);
        let record = serde_json::json!({"code": diagnostic.code, "source": location,
            "span": span, "message": diagnostic.to_string(), "status": "reported", "callers": [caller]});
        let step: String = step.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
        for category in ["error", "lint", "warning"] {
            fs::create_dir_all(self.root.join(&step).join(category)).map_err(|error| error.to_string())?;
        }
        self.store(&self.root.join(step).join("error"), key, record)?;
        for error in additional { self.report_diagnostic(error, caller, fallback)?; }
        Ok(())
    }

    pub(crate) fn scan(&self, manifest: Option<&Manifest>, input: &Path) -> Result<(Vec<PathBuf>, bool), String> {
        eprintln!("[build] Discovering source files: {}", input.display());
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
        let total = files.len();
        let mut last_progress = std::time::Instant::now();
        for (index, path) in files.into_iter().enumerate() {
            if index == 0 || index + 1 == total || last_progress.elapsed().as_secs() >= 1 {
                eprintln!("[build] Lexer/parser {}/{}: {}", index + 1, total, path.display());
                last_progress = std::time::Instant::now();
            }
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
        eprintln!("[build] Lexer/parser finished: {} of {} files parsed successfully", valid.len(), total);
        Ok((valid, failed))
    }

    pub(crate) fn failure(&self) -> String {
        format!("build failed; collected reports: {}", self.root.display())
    }

    pub(crate) fn compile_error(&self, source: &Path, error: &severian_driver::CompileError) -> Result<(), String> {
        use severian_driver::CompileError;
        if let CompileError::Diagnostic(diagnostic) = error {
            let fallback = match diagnostic.code {
                "E000101" | "E000102" | "E000103" => "lexer",
                "E000110" | "E000111" | "E000112" => "parser",
                _ => "semantic",
            };
            return self.report_diagnostic(diagnostic.clone(), source, fallback);
        }
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

pub(crate) fn owner_manifest(source: &Path) -> Option<PathBuf> {
    source.ancestors().skip(1).map(severian_driver::config::document::path)
        .find(|path| path.is_file())
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
    fn shared_diagnostic_keeps_its_actual_source_and_all_callers() {
        let root = std::env::temp_dir().join(format!("sev-shared-report-{}", std::process::id()));
        let reports = Reports::open(root.clone(), true).unwrap();
        let source = SourceFile::virtual_source("shared.sev", "invalid\n");
        let diagnostic = Diagnostic::new("E000112", "invalid declaration", Some(severian_source::Span::new(source.id, 0, 7)))
            .with_source(source);
        reports.compile_error(Path::new("first.sev"), &severian_driver::CompileError::Diagnostic(diagnostic.clone())).unwrap();
        reports.compile_error(Path::new("second.sev"), &severian_driver::CompileError::Diagnostic(diagnostic)).unwrap();
        let files = fs::read_dir(root.join("parser/error")).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(files.len(), 1);
        let report: serde_json::Value = serde_json::from_slice(&fs::read(files[0].path()).unwrap()).unwrap();
        assert_eq!(report["source"], "shared.sev");
        assert_eq!(report["callers"], serde_json::json!(["first.sev", "second.sev"]));
        fs::remove_dir_all(root).unwrap();
    }

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
