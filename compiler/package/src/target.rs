use super::*;

pub fn find_manifest(source: &Path) -> Option<PathBuf> {
    source.parent()?.ancestors().find_map(|directory| {
        let path = manifest_in(directory)?;
        parse_manifest(&path)
            .ok()
            .and_then(|manifest| manifest.get("package").is_some().then_some(path))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryTarget {
    pub name: String,
    pub source: PathBuf,
    pub package_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryTarget {
    pub name: String,
    pub source: PathBuf,
    pub artifact: PathBuf,
    pub manifest: PathBuf,
}

pub fn nearest_manifest(directory: &Path) -> Option<PathBuf> {
    directory.ancestors().find_map(manifest_in)
}

pub fn workspace_manifests(directory: &Path) -> Result<Vec<PathBuf>, PackageError> {
    let manifest_path = nearest_manifest(directory).ok_or_else(|| {
        PackageError::Manifest(format!(
            "could not find package.toml from {}",
            directory.display()
        ))
    })?;
    let manifest = parse_manifest(&manifest_path)?;
    let Some(members) = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
    else {
        return Ok(vec![manifest_path]);
    };
    let root = manifest_path
        .parent()
        .expect("a manifest path has a parent");
    members
        .iter()
        .map(|member| {
            let member = member.as_str().ok_or_else(|| {
                PackageError::Manifest("workspace members must be paths encoded as strings".into())
            })?;
            let directory = root.join(member);
            let path = manifest_in(&directory).unwrap_or_else(|| directory.join(MANIFEST_FILE));
            if !path.is_file() {
                return Err(PackageError::Manifest(format!(
                    "workspace member manifest {} does not exist",
                    path.display()
                )));
            }
            Ok(path)
        })
        .collect()
}

pub fn library_build_plan(manifest_path: &Path) -> Result<Vec<LibraryTarget>, PackageError> {
    let resolution = resolve_dependencies(manifest_path)?;
    let mut targets = Vec::new();
    for dependency in resolution.dependencies {
        let manifest = parse_manifest(&dependency.manifest)?;
        let source = dependency.root.join(library_path(&manifest));
        if manifest.get("lib").is_some() || source.is_file() {
            targets.push(LibraryTarget {
                name: dependency.import_name,
                source,
                artifact: library_artifact_path(&dependency.root, &dependency.package_name),
                manifest: dependency.manifest,
            });
        }
    }
    Ok(targets)
}

pub fn write_library_artifact(target: &LibraryTarget) -> Result<(), PackageError> {
    let source = std::fs::read_to_string(&target.source)?;
    if let Some(parent) = target.artifact.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &target.artifact,
        format!(
            "# severian-library-artifact v1\n# package {}\n{}",
            target.name,
            strip_package_tests(&source)
        ),
    )?;
    Ok(())
}

fn strip_package_tests(source: &str) -> String {
    let mut output = String::new();
    let mut skipped_test_indent = None;
    for line in source.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if let Some(test_indent) = skipped_test_indent {
            if trimmed.is_empty() || indent > test_indent {
                continue;
            }
            skipped_test_indent = None;
        }
        if (trimmed == "test:" || trimmed.starts_with("test ")) && trimmed.ends_with(':') {
            skipped_test_indent = Some(indent);
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
    output
}

/// Resolves the default binary target using Cargo-compatible `[package]` and
/// `[[bin]]` manifest fields.
pub fn default_binary_target(directory: &Path) -> Result<BinaryTarget, PackageError> {
    let direct = directory.join("main.sev");
    let manifest_path = nearest_manifest(directory);
    let Some(manifest_path) = manifest_path else {
        if direct.is_file() {
            return Ok(BinaryTarget {
                name: "main".into(),
                source: direct,
                package_root: directory.to_path_buf(),
            });
        }
        return Err(PackageError::Manifest(format!(
            "could not find `main.sev` or package.toml from {}",
            directory.display()
        )));
    };
    let manifest = parse_manifest(&manifest_path)?;
    let package_root = manifest_path
        .parent()
        .expect("a manifest path has a parent")
        .to_path_buf();
    if manifest.get("package").is_none() {
        let member = manifest
            .get("workspace")
            .and_then(toml::Value::as_table)
            .and_then(|workspace| workspace.get("members"))
            .and_then(toml::Value::as_array)
            .and_then(|members| members.first())
            .and_then(toml::Value::as_str)
            .ok_or_else(|| {
                PackageError::Manifest(format!(
                    "workspace manifest {} has no package or members",
                    manifest_path.display()
                ))
            })?;
        return default_binary_target(&package_root.join(member));
    }
    let binary = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .and_then(|binaries| binaries.first())
        .and_then(toml::Value::as_table);
    let binary_path = binary
        .and_then(|binary| binary.get("path"))
        .and_then(toml::Value::as_str)
        .unwrap_or("src/main.sev");
    let source = package_root.join(binary_path);
    if !source.is_file() {
        return Err(PackageError::Manifest(format!(
            "binary source {} does not exist",
            source.display()
        )));
    }
    let name = binary
        .and_then(|binary| binary.get("name"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            manifest
                .get("package")
                .and_then(toml::Value::as_table)
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "main".into());
    Ok(BinaryTarget {
        name,
        source,
        package_root,
    })
}

/// Resolves every source target built by `sev build` from the nearest package
/// or workspace manifest. This includes every `[[bin]]`, library-only packages,
/// and nested workspaces; selecting only the first binary would make the build
/// policy appear complete while silently omitting code.
pub fn workspace_binary_targets(directory: &Path) -> Result<Vec<BinaryTarget>, PackageError> {
    let direct = directory.join("main.sev");
    let manifest_path = nearest_manifest(directory);
    let Some(manifest_path) = manifest_path else {
        if direct.is_file() {
            return Ok(vec![default_binary_target(directory)?]);
        }
        return Err(PackageError::Manifest(format!(
            "could not find `main.sev` or package.toml from {}",
            directory.display()
        )));
    };
    let output_root = manifest_path
        .parent()
        .expect("a manifest path has a parent")
        .to_path_buf();
    let mut visited = BTreeSet::new();
    targets_from_manifest(&manifest_path, &output_root, &mut visited)
}

fn targets_from_manifest(
    manifest_path: &Path,
    output_root: &Path,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<Vec<BinaryTarget>, PackageError> {
    let manifest_path = manifest_path.to_path_buf();
    if !visited.insert(manifest_path.clone()) {
        return Err(PackageError::Manifest(format!(
            "workspace cycle includes {}",
            manifest_path.display()
        )));
    }
    let manifest = parse_manifest(&manifest_path)?;
    let root = manifest_path
        .parent()
        .expect("a manifest path has a parent");
    if manifest.get("package").is_some() {
        return package_build_targets(&manifest, root, output_root);
    }
    let members = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "workspace manifest {} has no members",
                manifest_path.display()
            ))
        })?;
    let mut targets = Vec::new();
    for member in members {
        let member = member.as_str().ok_or_else(|| {
            PackageError::Manifest("workspace members must be paths encoded as strings".into())
        })?;
        let member_root = root.join(member);
        let member_manifest_path =
            manifest_in(&member_root).unwrap_or_else(|| member_root.join(MANIFEST_FILE));
        targets.extend(targets_from_manifest(
            &member_manifest_path,
            output_root,
            visited,
        )?);
    }
    Ok(targets)
}

fn package_build_targets(
    manifest: &toml::Value,
    root: &Path,
    output_root: &Path,
) -> Result<Vec<BinaryTarget>, PackageError> {
    let package_name = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        .unwrap_or("main");
    let mut targets = Vec::new();
    if let Some(binaries) = manifest.get("bin").and_then(toml::Value::as_array) {
        for (index, binary) in binaries.iter().enumerate() {
            let binary = binary.as_table().ok_or_else(|| {
                PackageError::Manifest("every `[[bin]]` entry must be a table".into())
            })?;
            let path = binary
                .get("path")
                .and_then(toml::Value::as_str)
                .unwrap_or("src/main.sev");
            let source = root.join(path);
            if !source.is_file() {
                return Err(PackageError::Manifest(format!(
                    "binary source {} does not exist",
                    source.display()
                )));
            }
            let name = binary
                .get("name")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{package_name}-{index}"));
            targets.push(BinaryTarget {
                name,
                source,
                package_root: output_root.to_path_buf(),
            });
        }
    } else {
        let main = [root.join("src/main.sev"), root.join("main.sev")]
            .into_iter()
            .find(|path| path.is_file());
        if let Some(source) = main {
            targets.push(BinaryTarget {
                name: package_name.to_owned(),
                source,
                package_root: output_root.to_path_buf(),
            });
        }
    }

    let library = manifest
        .get("lib")
        .and_then(toml::Value::as_table)
        .and_then(|library| library.get("path"))
        .and_then(toml::Value::as_str)
        .map(|path| root.join(path))
        .or_else(|| {
            root.join("src/lib.sev")
                .is_file()
                .then(|| root.join("src/lib.sev"))
        });
    if let Some(source) = library {
        if !source.is_file() {
            return Err(PackageError::Manifest(format!(
                "library source {} does not exist",
                source.display()
            )));
        }
        let name = if root == output_root {
            source
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("lib")
                .to_owned()
        } else {
            format!("{package_name}-lib")
        };
        targets.push(BinaryTarget {
            name,
            source,
            package_root: output_root.to_path_buf(),
        });
    }
    Ok(targets)
}

/// Resolves the first binary target for `sev build` from the nearest manifest.
pub fn default_binary_source(directory: &Path) -> Result<PathBuf, PackageError> {
    Ok(default_binary_target(directory)?.source)
}

pub fn load_path_dependency_sources(manifest_path: &Path) -> Result<Vec<String>, PackageError> {
    resolve_dependencies(manifest_path)?
        .dependencies
        .into_iter()
        .map(|dependency| {
            let manifest = parse_manifest(&dependency.manifest)?;
            let source = dependency.root.join(library_path(&manifest));
            std::fs::read_to_string(&source).map_err(|error| {
                PackageError::Manifest(format!(
                    "could not read library for `{}` at {}: {error}",
                    dependency.import_name,
                    source.display()
                ))
            })
        })
        .collect()
}
