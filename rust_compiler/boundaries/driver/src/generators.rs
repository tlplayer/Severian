//! Declared package recipes run before source discovery and cache validation.
use severian_driver::config::{Catalog, Manifest};
use std::process::Command;

pub(crate) fn prepare(manifest: &Manifest) -> Result<(), String> {
    if std::env::var_os("SEVERIAN_ACTIVE_GENERATOR").is_some_and(|value| !value.is_empty()) {
        return Ok(());
    }
    for package in manifest.package_graph.packages.values() {
        let Some(value) = package.manifest.get("build").and_then(|build| build.get("generators")) else { continue; };
        Catalog::load()?.validate("build.generators", &value.to_string())?;
        for recipe in value.as_array().ok_or("build.generators requires an array")? {
            let recipe = package.root.join(recipe.as_str().ok_or("generator path must be a string")?);
            let recipe = recipe.canonicalize().map_err(|error| format!("{}: {error}", recipe.display()))?;
            let root = package.root.canonicalize().map_err(|error| error.to_string())?;
            if !recipe.starts_with(&root) {
                return Err("generator path escapes its package".into());
            }
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
            let status = command.arg(&recipe).status()
                .map_err(|error| format!("{}: {error}", recipe.display()))?;
            if !status.success() {
                return Err(format!("package generator failed: {} ({status})", recipe.display()));
            }
        }
    }
    Ok(())
}
