//! Declared package recipes run before source discovery and cache validation.
use severian_driver::config::{Catalog, Manifest};
use std::process::Command;
use std::collections::BTreeSet;
use crate::build_reports::Reports;

pub(crate) struct Prepared {
    pub(crate) roots: BTreeSet<std::path::PathBuf>,
    pub(crate) packages: BTreeSet<std::path::PathBuf>,
    pub(crate) failed: bool,
}

pub(crate) fn prepare_reachable(
    compiler: &severian_driver::Compiler,
    manifest: Option<&Manifest>,
    sources: &[std::path::PathBuf],
    reports: Option<&Reports>,
) -> Result<Prepared, String> {
    let mut roots = sources.iter().cloned().collect::<BTreeSet<_>>();
    if std::env::var_os("SEVERIAN_ACTIVE_GENERATOR").is_some_and(|value| !value.is_empty()) {
        return Ok(Prepared { roots, packages: BTreeSet::new(), failed: false });
    }
    let mut prepared = BTreeSet::new();
    let mut required_packages = BTreeSet::new();
    let mut generator_failed = false;
    // Discover before running recipes, then repeat after each wave: generated
    // sources may add imports. Failed provisional edges are reported only after
    // their owning generators have had the opportunity to produce the source.
    loop {
        let before = (roots.len(), required_packages.len());
        for source in roots.clone() {
            let discovery = compiler.discover_modules(&source).map_err(|error| error.to_string())?;
            for path in discovery.attempted.iter().chain(discovery.graph.modules.iter().map(|module| &module.path)) {
                if let Some(owner) = severian_driver::build_reports::owner_manifest(path) {
                    if let Some(root) = owner.parent().and_then(|path| path.canonicalize().ok()) {
                        required_packages.insert(root);
                    }
                }
            }
        }
        let mut pending = BTreeSet::new();
        if let Some(manifest) = manifest {
            for package in manifest.package_graph.packages.values() {
                if !required_packages.contains(&package.root) || prepared.contains(&package.root) { continue; }
                pending.insert(package.root.clone());
                if let Some(recipes) = package.manifest.get("build").and_then(|build| build.get("generators")).and_then(toml::Value::as_array) {
                    for recipe in recipes.iter().filter_map(toml::Value::as_str) {
                        let recipe = package.root.join(recipe);
                        if recipe.extension().is_some_and(|extension| extension == "sev") { roots.insert(recipe); }
                    }
                }
            }
            // Reach a fixed point over recipe imports before executing any
            // recipe, then run dependency packages before their consumers.
            if before != (roots.len(), required_packages.len()) { continue; }
            if !pending.is_empty() {
                let selected = pending.iter().filter(|root| {
                    let package = manifest.package_graph.packages.values().find(|package| &package.root == *root).unwrap();
                    !package.dependencies.values().any(|id| pending.contains(&manifest.package_graph.packages[id].root))
                }).cloned().collect::<BTreeSet<_>>();
                if selected.is_empty() { return Err("reachable generator packages contain a dependency cycle".into()); }
                if let Err(error) = prepare_reported(manifest, reports, Some(&selected)) {
                    if let Some(reports) = reports {
                        reports.record("generator", "error", &manifest.root, &error)?;
                        generator_failed = true;
                    } else { return Err(error); }
                }
                prepared.extend(selected);
                continue;
            }
        }
        break;
    }
    Ok(Prepared { roots, packages: required_packages, failed: generator_failed })
}

pub(crate) fn prepare_reported(manifest: &Manifest, reports: Option<&Reports>, selected: Option<&BTreeSet<std::path::PathBuf>>) -> Result<(), String> {
    if std::env::var_os("SEVERIAN_ACTIVE_GENERATOR").is_some_and(|value| !value.is_empty()) {
        return Ok(());
    }
    let mut failures = Vec::new();
    let mut reported_stderr = BTreeSet::new();
    for package in manifest.package_graph.packages.values() {
        if selected.is_some_and(|selected| !selected.contains(&package.root)) { continue; }
        let result = (|| -> Result<(), String> {
            let Some(value) = package.manifest.get("build").and_then(|build| build.get("generators")) else { return Ok(()); };
            Catalog::load()?.validate("build.generators", &value.to_string())?;
            for recipe in value.as_array().ok_or("build.generators requires an array")? {
                let result = (|| -> Result<(), String> {
                    let recipe = package.root.join(recipe.as_str().ok_or("generator path must be a string")?);
                    let recipe = recipe.canonicalize().map_err(|error| format!("{}: {error}", recipe.display()))?;
                    let root = package.root.canonicalize().map_err(|error| error.to_string())?;
                    if !recipe.starts_with(&root) { return Err("generator path escapes its package".into()); }
                    let mut command = match recipe.extension().and_then(|value| value.to_str()) {
                        Some("sev") => {
                            let compiler = match std::env::var_os("SEVERIAN_GENERATOR_COMPILER") {
                                Some(value) if !value.is_empty() => value,
                                _ => std::env::current_exe().map_err(|error| error.to_string())?.into_os_string(),
                            };
                            let mut command = Command::new(compiler);
                            command.env("SEVERIAN_ACTIVE_GENERATOR", &recipe);
                            command
                        }
                        Some("py") => Command::new("python3"),
                        _ => return Err(format!("unsupported build recipe: {}", recipe.display())),
                    };
                    if let Some(reports) = reports { command.env(crate::build_reports::ENV, &reports.root); }
                    let output = command.arg(&recipe).output().map_err(|error| format!("{}: {error}", recipe.display()))?;
                    if reports.is_none() { print!("{}", String::from_utf8_lossy(&output.stdout)); }
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if reports.is_none() && should_print_stderr(&stderr, output.status.success(), &mut reported_stderr) {
                        eprint!("{stderr}");
                    }
                    if let Some(reports) = reports {
                        reports.record("generator", "trace", &recipe, &format!("status: {}\nstdout:\n{}\nstderr:\n{}", output.status, String::from_utf8_lossy(&output.stdout), stderr))?;
                        for warning in stderr.lines().filter(|line| line.trim_start().starts_with("warning:")) {
                            reports.record("generator", "warning", &recipe, warning)?;
                        }
                        if !output.status.success() {
                            let status = if stderr.lines().any(|line| line.starts_with("error: E")) {
                                "blocked: generator compilation failed"
                            } else { "generator execution failed" };
                            reports.record("generator", "error", &recipe, &format!("{status}: {}\n{}\n{}", output.status, String::from_utf8_lossy(&output.stdout), stderr))?;
                        }
                    }
                    if !output.status.success() { return Err(format!("package generator failed: {} ({})", recipe.display(), output.status)); }
                    Ok(())
                })();
                if let Err(error) = result {
                    if reports.is_none() { return Err(error); }
                    failures.push(error);
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            if reports.is_none() { return Err(error); }
            failures.push(error);
        }
    }
    if failures.is_empty() { Ok(()) } else { Err(failures.join("\n")) }
}

// Keep every recipe's report, but print an identical compiler failure once.
// Successful recipe output and non-compiler failures remain untouched.
fn should_print_stderr(stderr: &str, success: bool, seen: &mut BTreeSet<String>) -> bool {
    success || !stderr.lines().any(|line| line.starts_with("error: E"))
        || seen.insert(stderr.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_generator_failure_is_printed_once_without_hiding_distinct_errors() {
        let mut seen = BTreeSet::new();
        let first = "error: E000211: invalid assignment\n--> shared.sev:11:9\n";
        assert!(should_print_stderr(first, false, &mut seen));
        assert!(!should_print_stderr(first, false, &mut seen));
        assert!(should_print_stderr("error: E000211: invalid assignment\n--> other.sev:11:9\n", false, &mut seen));
        assert!(should_print_stderr(first, true, &mut seen));
        for _ in 0..2 {
            assert!(should_print_stderr("generator execution failed\n", false, &mut seen));
        }
    }
}
