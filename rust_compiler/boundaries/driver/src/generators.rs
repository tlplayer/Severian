//! Declared package recipes run before source discovery and cache validation.
use severian_driver::config::{Catalog, Manifest};
use std::process::Command;

pub(crate) fn prepare(manifest: &Manifest) -> Result<(), String> {
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
            let status = Command::new("python3").arg(&recipe).status()
                .map_err(|error| format!("{}: {error}", recipe.display()))?;
            if !status.success() {
                return Err(format!("package generator failed: {} ({status})", recipe.display()));
            }
        }
    }
    Ok(())
}
