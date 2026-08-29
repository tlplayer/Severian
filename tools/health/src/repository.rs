use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn discover(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(root) = explicit {
        return canonical_root(&root);
    }
    let cwd = env::current_dir().map_err(|error| error.to_string())?;
    for ancestor in cwd.ancestors() {
        if ancestor.join("Cargo.toml").is_file() && ancestor.join("compiler").is_dir() {
            return canonical_root(ancestor);
        }
    }
    Err("could not find the Severian repository root".into())
}

fn canonical_root(root: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(root).map_err(|error| format!("could not resolve {}: {error}", root.display()))
}

pub fn rust_sources(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    collect(root, root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect(root: &Path, directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            if ignored(relative) {
                continue;
            }
            collect(root, &path, output)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            output.push(relative.to_path_buf());
        }
    }
    Ok(())
}

fn ignored(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".git" | "target" | "third_party" | ".codex" | ".agents")
        )
    })
}

pub fn changed_paths(root: &Path, base: &str) -> Result<BTreeSet<PathBuf>, String> {
    let mut paths = BTreeSet::new();
    for arguments in [
        vec!["diff", "--name-only", &format!("{base}...HEAD")],
        vec!["diff", "--name-only"],
        vec!["ls-files", "--others", "--exclude-standard"],
    ] {
        let output = Command::new("git")
            .args(&arguments)
            .current_dir(root)
            .output()
            .map_err(|error| format!("could not run git: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git {} failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        paths.extend(
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| !line.is_empty())
                .map(PathBuf::from),
        );
    }
    Ok(paths)
}

pub fn changed_lines(
    root: &Path,
    base: &str,
) -> Result<BTreeMap<PathBuf, BTreeSet<usize>>, String> {
    let mut changed = BTreeMap::new();
    for arguments in [
        vec!["diff", "--unified=0", &format!("{base}...HEAD")],
        vec!["diff", "--unified=0"],
    ] {
        let output = Command::new("git")
            .args(&arguments)
            .current_dir(root)
            .output()
            .map_err(|error| format!("could not run git: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git {} failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        parse_changed_lines(&String::from_utf8_lossy(&output.stdout), &mut changed);
    }
    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not run git: {error}"))?;
    if !untracked.status.success() {
        return Err("git ls-files failed while collecting changed lines".into());
    }
    for path in String::from_utf8_lossy(&untracked.stdout)
        .lines()
        .map(PathBuf::from)
    {
        let Ok(source) = fs::read_to_string(root.join(&path)) else {
            continue;
        };
        changed
            .entry(path)
            .or_insert_with(BTreeSet::new)
            .extend(1..=source.lines().count());
    }
    Ok(changed)
}

fn parse_changed_lines(source: &str, output: &mut BTreeMap<PathBuf, BTreeSet<usize>>) {
    let mut path = None;
    for line in source.lines() {
        if let Some(value) = line.strip_prefix("+++ b/") {
            path = Some(PathBuf::from(value));
            continue;
        }
        let (Some(path), Some(hunk)) = (&path, line.strip_prefix("@@ ")) else {
            continue;
        };
        let Some(added) = hunk.split_whitespace().find(|part| part.starts_with('+')) else {
            continue;
        };
        let mut range = added.trim_start_matches('+').split(',');
        let Some(start) = range.next().and_then(|value| value.parse::<usize>().ok()) else {
            continue;
        };
        let count = range
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        output
            .entry(path.clone())
            .or_default()
            .extend(start..start.saturating_add(count));
    }
}

pub fn churn_90d(root: &Path) -> BTreeSet<PathBuf> {
    let Ok(output) = Command::new("git")
        .args(["log", "--since=90.days", "--name-only", "--pretty=format:"])
        .current_dir(root)
        .output()
    else {
        return BTreeSet::new();
    };
    if !output.status.success() {
        return BTreeSet::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.ends_with(".rs"))
        .map(PathBuf::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_added_hunk_ranges() {
        let mut changed = BTreeMap::new();
        parse_changed_lines(
            "+++ b/compiler/a.rs\n@@ -4,2 +7,3 @@\n+x\n+y\n+z\n",
            &mut changed,
        );
        assert_eq!(
            changed.get(Path::new("compiler/a.rs")),
            Some(&BTreeSet::from([7, 8, 9]))
        );
    }
}
