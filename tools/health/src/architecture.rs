use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub fn is_user_input_path(path: &Path) -> bool {
    [
        "compiler/source",
        "compiler/frontend/lexer",
        "compiler/frontend/parser",
        "compiler/frontend/modules",
        "compiler/frontend/semantic",
    ]
    .into_iter()
    .any(|prefix| path.starts_with(prefix))
}

pub fn is_artifact_path(path: &Path) -> bool {
    ["artifact", "package", "interface", "pipeline", "backend"]
        .into_iter()
        .any(|part| path.to_string_lossy().contains(part))
}

pub fn cargo_manifests(root: &Path) -> Result<Vec<PathBuf>, String> {
    fn visit(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let file_type = entry.file_type().map_err(|error| error.to_string())?;
            let path = entry.path();
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                if matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some(".git" | "target" | "third_party")
                ) {
                    continue;
                }
                visit(&path, output)?;
            } else if path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
                output.push(path);
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    visit(root, &mut output)?;
    Ok(output)
}

pub fn package_name(source: &str) -> Option<String> {
    let mut package = false;
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            package = line == "[package]";
        } else if package && line.starts_with("name") {
            return quoted_values(line).into_iter().next();
        }
    }
    None
}

pub fn severian_dependencies(source: &str) -> BTreeSet<String> {
    let mut dependencies = false;
    let mut output = BTreeSet::new();
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            dependencies = line.ends_with("dependencies]");
        } else if dependencies {
            if let Some((name, _)) = line.split_once('=') {
                let name = name.trim();
                if name.starts_with("severian-") {
                    output.insert(name.into());
                }
            }
        }
    }
    output
}

pub fn parse_architecture_allow(source: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut active = false;
    let mut current = None::<String>;
    let mut output = BTreeMap::<String, BTreeSet<String>>::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            active = trimmed == "[workspace.metadata.architecture.allow]";
            current = None;
            continue;
        }
        if !active || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some((name, suffix)) = trimmed.split_once('=') {
            current = Some(name.trim().to_string());
            output
                .entry(name.trim().to_string())
                .or_default()
                .extend(quoted_values(suffix));
            if suffix.contains(']') {
                current = None;
            }
        } else if let Some(name) = &current {
            output
                .entry(name.clone())
                .or_default()
                .extend(quoted_values(trimmed));
            if trimmed.contains(']') {
                current = None;
            }
        }
    }
    output
}

fn quoted_values(source: &str) -> Vec<String> {
    source
        .split('"')
        .enumerate()
        .filter(|(index, _)| index % 2 == 1)
        .map(|(_, value)| value.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_allow_supports_multiline_arrays() {
        let values = parse_architecture_allow(
            "[workspace.metadata.architecture.allow]\na = [\n \"b\",\n \"c\",\n]\n",
        );
        assert_eq!(
            values["a"],
            ["b", "c"].into_iter().map(str::to_string).collect()
        );
    }
}
