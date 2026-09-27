//! Declared package recipes run before source discovery and cache validation.
use severian_driver::config::{Catalog, Manifest};
use std::process::Command;
use crate::build_reports::Reports;

pub(crate) fn prepare(manifest: &Manifest) -> Result<(), String> {
    prepare_reported(manifest, None)
}

pub(crate) fn prepare_reported(manifest: &Manifest, reports: Option<&Reports>) -> Result<(), String> {
    if std::env::var_os("SEVERIAN_ACTIVE_GENERATOR").is_some_and(|value| !value.is_empty()) {
        return Ok(());
    }
    let mut failures = Vec::new();
    for package in manifest.package_graph.packages.values() {
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
                    print!("{}", String::from_utf8_lossy(&output.stdout));
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprint!("{stderr}");
                    if let Some(reports) = reports {
                        for warning in stderr.lines().filter(|line| line.trim_start().starts_with("warning:")) {
                            reports.record("generator", "warning", &recipe, warning)?;
                        }
                        if !output.status.success() {
                            reports.record("generator", "error", &recipe, &format!("{}\n{}\n{}", output.status, String::from_utf8_lossy(&output.stdout), stderr))?;
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
