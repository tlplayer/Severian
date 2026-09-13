//! Content-checked freshness for the bootstrap CLI, before semantic analysis.
use serde::{Deserialize, Serialize};
use severian_driver::Compiler;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Input {
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Snapshot {
    configuration: String,
    tools: Vec<String>,
    inputs: Vec<Input>,
}

#[derive(Serialize, Deserialize)]
struct Record {
    schema_version: u32,
    roots: BTreeSet<PathBuf>,
    snapshot: Snapshot,
    output: Input,
}

fn files(root: &Path, excluded: &Path, found: &mut BTreeSet<PathBuf>, seen: &mut BTreeSet<PathBuf>) -> Result<(), String> {
    let root = fs::canonicalize(root).map_err(|e| format!("{}: {e}", root.display()))?;
    if root == excluded || !seen.insert(root.clone()) { return Ok(()); }
    if root.is_file() { found.insert(root); return Ok(()); }
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(name.as_ref(), "package.pkg" | ".git" | "target" | "__pycache__" | ".pytest_cache") || name.starts_with(".sev-") { continue; }
        files(&entry.path(), excluded, found, seen)?;
    }
    Ok(())
}

// One hashing process per batch, not one fork/exec for every source file.
fn hashes(paths: &[PathBuf]) -> Result<Vec<Input>, String> {
    let mut result = Vec::new();
    for batch in paths.chunks(128) {
        let output = Command::new("sha256sum").args(["--zero", "--"]).args(batch).output().map_err(|e| e.to_string())?;
        if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).into_owned()); }
        let records = output.stdout.split(|b| *b == 0).filter(|r| !r.is_empty()).collect::<Vec<_>>();
        if records.len() != batch.len() { return Err("invalid source hash inventory".into()); }
        for (path, record) in batch.iter().zip(records) {
            if record.len() < 66 || !record[..64].iter().all(u8::is_ascii_hexdigit) { return Err("invalid SHA-256 output".into()); }
            result.push(Input { path: path.clone(), sha256: String::from_utf8(record[..64].to_vec()).map_err(|e| e.to_string())? });
        }
    }
    Ok(result)
}

fn tools() -> Result<Vec<String>, String> {
    let mut identities = Vec::new();
    for (variable, default) in [("SEVERIAN_CLANG", "clang-21"), ("SEVERIAN_MLIR_OPT", "mlir-opt-21"), ("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21"), ("SEVERIAN_LINKER", "ld.lld-21")] {
        let executable = std::env::var(variable).unwrap_or_else(|_| default.into());
        let output = Command::new(&executable).arg("--version").output().map_err(|e| format!("{executable}: {e}"))?;
        if !output.status.success() { return Err(format!("could not identify {executable}")); }
        identities.push(format!("{variable}={executable}\n{}", String::from_utf8_lossy(&output.stdout)));
    }
    Ok(identities)
}

fn snapshot(configuration: &str, roots: &BTreeSet<PathBuf>, output: &Path) -> Result<Snapshot, String> {
    let mut inputs = BTreeSet::new();
    let mut seen = BTreeSet::new();
    for root in roots { files(root, output, &mut inputs, &mut seen)?; }
    inputs.insert(std::env::current_exe().map_err(|e| e.to_string())?);
    Ok(Snapshot { configuration: configuration.into(), tools: tools()?, inputs: hashes(&inputs.into_iter().collect::<Vec<_>>())? })
}

fn package_root(source: &Path) -> PathBuf {
    for directory in source.ancestors().skip(1) {
        if severian_driver::config::document::path(&directory).is_file() { return directory.into(); }
    }
    source.parent().expect("source parent").into()
}

pub(crate) fn compile(compiler: &Compiler, source: &Path, output: &Path, root: &Path, configuration: String, declared: Vec<PathBuf>) -> Result<bool, String> {
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let output = fs::canonicalize(parent).map_err(|e| e.to_string())?.join(output.file_name().ok_or("output has no filename")?);
    // This hash names a lock only. Reuse always compares SHA-256 input/output
    // inventories and the complete configuration, never this short identifier.
    let key = output.as_os_str().as_encoded_bytes().iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3));
    let directory = root.join("package.pkg/build/bootstrap");
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let lock = File::create(directory.join(format!("{key:016x}.lock"))).map_err(|e| e.to_string())?;
    lock.lock().map_err(|e| e.to_string())?;
    let record_path = directory.join(format!("{key:016x}.json"));
    let previous = fs::read(&record_path).ok().and_then(|bytes| serde_json::from_slice::<Record>(&bytes).ok()).filter(|r| r.schema_version == 1);
    let mut roots = declared.into_iter().map(|p| fs::canonicalize(p).map_err(|e| e.to_string())).collect::<Result<BTreeSet<_>, _>>()?;
    if let Some(record) = &previous { roots.extend(record.roots.iter().filter(|p| p.exists()).cloned()); }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).expect("driver repository");
    for directory in ["rust_compiler/runtime/native", "library/core/memory", "library/system/extern", "sev_compiler/universal"] { roots.insert(repository.join(directory)); }
    let before = snapshot(&configuration, &roots, &output)?;
    let force = std::env::var("SEVERIAN_FORCE_REBUILD").as_deref() == Ok("1");
    if !force && output.is_file() {
        if let Some(record) = previous {
            if record.snapshot == before && hashes(std::slice::from_ref(&output))?[0] == record.output {
                println!("fresh {}", output.display());
                return Ok(false);
            }
        }
    }
    // Discover new import/provider roots only on a miss. Warm checks do no
    // parsing, semantic analysis, lowering or native code generation.
    for module in compiler.resolved_module_paths(source).map_err(|e| e.to_string())? { roots.insert(package_root(&module)); }
    let before = snapshot(&configuration, &roots, &output)?;
    let staging = parent.join(format!(".sev-build-{}-{key:016x}", std::process::id()));
    if let Err(error) = compiler.compile_file(source, &staging) {
        let _ = fs::remove_file(&staging);
        return Err(error.to_string());
    }
    if snapshot(&configuration, &roots, &output)? != before {
        let _ = fs::remove_file(&staging);
        return Err("compiler inputs changed during compilation; no fresh result recorded".into());
    }
    fs::rename(&staging, &output).map_err(|e| e.to_string())?;
    let record = Record { schema_version: 1, roots, snapshot: before, output: hashes(std::slice::from_ref(&output))?.remove(0) };
    let staging = directory.join(format!(".sev-record-{}-{key:016x}.json", std::process::id()));
    fs::write(&staging, serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fs::rename(staging, record_path).map_err(|e| e.to_string())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inventory_detects_edits_additions_and_deletions_but_ignores_build_outputs() {
        let root = std::env::temp_dir().join(format!("sev-freshness-{}", std::process::id()));
        fs::create_dir_all(root.join("package.pkg")).unwrap();
        let source = root.join("input.sev");
        fs::write(&source, "one").unwrap();
        let inventory = || { let mut found = BTreeSet::new(); files(&root, &root.join("program"), &mut found, &mut BTreeSet::new()).unwrap(); hashes(&found.into_iter().collect::<Vec<_>>()).unwrap() };
        let initial = inventory();
        fs::write(root.join("package.pkg/output"), "generated").unwrap();
        assert_eq!(initial, inventory());
        fs::write(&source, "two").unwrap();
        assert_ne!(initial, inventory());
        fs::write(&source, "one").unwrap();
        assert_eq!(initial, inventory());
        fs::write(root.join("added.sev"), "new").unwrap();
        assert_ne!(initial, inventory());
        fs::remove_file(source).unwrap();
        assert_ne!(initial, inventory());
        fs::remove_dir_all(root).unwrap();
    }
}
