//! Disposable reports shared by an update and its child compiler invocations.
use crate::config::Manifest;
use severian_diagnostics::{Diagnostic, DiagnosticContext};
use severian_source::SourceFile;
use std::collections::{BTreeMap, BTreeSet};
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::hash::{Hash, Hasher};

thread_local! {
    static ACTIVE_ROOT: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

pub fn warning(source: &Path, message: &str) -> Result<(), String> {
    let root = ACTIVE_ROOT.with(|root| root.borrow().clone())
        .or_else(|| std::env::var_os(ENV).filter(|value| !value.is_empty()).map(PathBuf::from));
    if let Some(root) = root {
        Reports { root }.record("dependencies", "warning", source, message)
    } else {
        eprintln!("warning: {message}");
        Ok(())
    }
}

pub const ENV: &str = "SEVERIAN_BUILD_REPORT_DIR";

pub struct Reports {
    pub root: PathBuf,
}

impl Reports {
    pub fn begin() -> Result<Self, String> {
        let inherited = std::env::var_os(ENV).filter(|value| !value.is_empty());
        let root = match &inherited {
            Some(root) => PathBuf::from(root),
            None => std::env::current_dir().map_err(|error| error.to_string())?.join("package.pkg/debug/build"),
        };
        let reports = Self::open(root, inherited.is_none())?;
        ACTIVE_ROOT.with(|root| *root.borrow_mut() = Some(reports.root.clone()));
        Ok(reports)
    }

    fn open(root: PathBuf, reset: bool) -> Result<Self, String> {
        if reset && root.exists() {
            fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
        }
        fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        if reset || !root.join("log.txt").exists() {
            fs::write(root.join("log.txt"), "Build trace (grouped by source file)\n").map_err(|error| error.to_string())?;
        }
        Ok(Self { root })
    }

    pub fn record(&self, step: &str, category: &str, source: &Path, message: &str) -> Result<(), String> {
        if !matches!(category, "error" | "lint" | "warning" | "trace") {
            return Err(format!("invalid build report category: {category}"));
        }
        let step = if step.trim().is_empty() { "build" } else { step };
        let step: String = step.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
        let directory = self.root.join(step);
        for category in ["error", "lint", "warning", "trace"] {
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
        for category in ["error", "lint", "warning", "trace"] {
            fs::create_dir_all(self.root.join(&step).join(category)).map_err(|error| error.to_string())?;
        }
        self.store(&self.root.join(step).join("error"), key, record)?;
        for error in additional { self.report_diagnostic(error, caller, fallback)?; }
        Ok(())
    }

    pub fn scan(&self, manifest: Option<&Manifest>, input: &Path) -> Result<(Vec<PathBuf>, bool), String> {
        self.record("build", "trace", input, "Discovering source files")?;
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
                Ok(_) => {
                    self.record("parser", "trace", &path, "Source parsed successfully")?;
                    valid.push(path);
                }
                Err(error) => {
                    self.diagnostic(error, &source, "parser")?;
                    failed = true;
                }
            }
        }
        self.record("build", "trace", input, &format!("Lexer/parser: {} of {total} files parsed successfully", valid.len()))?;
        Ok((valid, failed))
    }

    pub fn finish(&self) -> Result<(), String> {
        fn read(directory: &Path, groups: &mut BTreeMap<String, Vec<(String, serde_json::Value)>>) -> Result<(), String> {
            for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
                let path = entry.map_err(|error| error.to_string())?.path();
                if path.is_dir() { read(&path, groups)?; }
                else if path.extension().is_some_and(|extension| extension == "json") {
                    let record: serde_json::Value = serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
                        .map_err(|error| error.to_string())?;
                    let source = record["source"].as_str().unwrap_or("build/update").to_owned();
                    let category = path.parent().and_then(Path::file_name).unwrap().to_string_lossy();
                    let stage = path.parent().and_then(Path::parent).and_then(Path::file_name).unwrap().to_string_lossy();
                    groups.entry(source).or_default().push((format!("{stage}/{category}"), record));
                }
            }
            Ok(())
        }
        let mut groups = BTreeMap::new();
        read(&self.root, &mut groups)?;
        let mut output = String::from("Build trace (grouped by source file)\n");
        for (source, mut records) in groups {
            output.push_str(&format!("\n===== {source} =====\n"));
            records.sort_by_key(|(stage, record)| (stage.clone(), record["span"][0].as_u64().unwrap_or(0), record["message"].as_str().unwrap_or("").to_owned()));
            for (stage, record) in records {
                output.push_str(&format!("\n[{stage}]\n{}\n", record["message"].as_str().unwrap_or("")));
                if let Some(callers) = record["callers"].as_array() {
                    for caller in callers {
                        if let Some(caller) = caller.as_str().filter(|caller| *caller != source) {
                            output.push_str(&format!("  required by: {caller}\n"));
                        }
                    }
                }
            }
        }
        fs::write(self.root.join("log.txt"), output).map_err(|error| error.to_string())
    }

    pub fn failure(&self) -> String {
        format!("build failed; see {}", self.root.join("log.txt").display())
    }

    pub fn compile_error(&self, source: &Path, error: &crate::CompileError) -> Result<(), String> {
        use crate::CompileError;
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

pub fn owner_manifest(source: &Path) -> Option<PathBuf> {
    source.ancestors().skip(1).map(crate::config::document::path)
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
        reports.compile_error(Path::new("first.sev"), &crate::CompileError::Diagnostic(diagnostic.clone())).unwrap();
        reports.compile_error(Path::new("second.sev"), &crate::CompileError::Diagnostic(diagnostic)).unwrap();
        let files = fs::read_dir(root.join("parser/error")).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(files.len(), 1);
        let report: serde_json::Value = serde_json::from_slice(&fs::read(files[0].path()).unwrap()).unwrap();
        assert_eq!(report["source"], "shared.sev");
        assert_eq!(report["callers"], serde_json::json!(["first.sev", "second.sev"]));
        reports.record("dependencies", "warning", Path::new("package.json"), "missing optional package").unwrap();
        reports.finish().unwrap();
        let log = fs::read_to_string(root.join("log.txt")).unwrap();
        assert_eq!(log.matches("===== shared.sev =====").count(), 1);
        assert_eq!(log.matches("E000112: invalid declaration").count(), 1);
        assert!(log.contains("required by: first.sev") && log.contains("required by: second.sev"));
        assert!(log.contains("[dependencies/warning]\nmissing optional package"));
        Reports::open(root.clone(), true).unwrap();
        assert!(!fs::read_to_string(root.join("log.txt")).unwrap().contains("invalid declaration"));
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
