use crate::{
    install::preserve_external_lock,
    lockfile::{self, LockedPackage},
    manifest_in, package_name, parse_manifest,
    sandbox::validate_package_payload,
    PackageError, MANIFEST_FILE,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const LOCK_FILE: &str = "sev.lock";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDependency {
    /// The name source code imports. This is the dependency table key and can
    /// deliberately differ from the published package name.
    pub import_name: String,
    pub package_name: String,
    pub version: String,
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub source: String,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resolution {
    pub dependencies: Vec<ResolvedDependency>,
    pub lockfile: PathBuf,
}

/// Publishes an immutable package version to an on-disk Severian registry.
/// The registry keeps package storage separate from the consumer cache and
/// records a trusted digest for every published version.
pub fn publish_package(
    manifest_path: &Path,
    registry: Option<&Path>,
) -> Result<ResolvedDependency, PackageError> {
    let manifest_path = manifest_path.canonicalize()?;
    let root = manifest_path
        .parent()
        .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
    let manifest = parse_manifest(&manifest_path)?;
    validate_package_payload(root)?;
    let name = package_name(&manifest, &manifest_path)?.to_owned();
    validate_package_name(&name)?;
    let version = package_version(&manifest, &manifest_path)?;
    parse_version(&version).ok_or_else(|| {
        PackageError::Manifest(format!("package `{name}` has invalid version `{version}`"))
    })?;
    let registry = registry
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("SEVERIAN_REGISTRY").map(PathBuf::from))
        .unwrap_or_else(default_registry_root);
    let registry = strip_file_scheme(&registry)?;
    let destination = registry.join("packages").join(&name).join(&version);
    if destination.exists() {
        return Err(PackageError::Manifest(format!(
            "package `{name}` {version} is already published in {}",
            registry.display()
        )));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| PackageError::Manifest("registry package path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{version}.{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    copy_tree(root, &temporary)?;
    let checksum = directory_checksum(&temporary)?;
    let checksum_path = registry
        .join("checksums")
        .join(&name)
        .join(format!("{version}.sha256"));
    if let Some(parent) = checksum_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut checksum_file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&checksum_path)
    {
        Ok(file) => file,
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error.into());
        }
    };
    if let Err(error) = checksum_file.write_all(format!("{checksum}\n").as_bytes()) {
        drop(checksum_file);
        let _ = fs::remove_file(&checksum_path);
        let _ = fs::remove_dir_all(&temporary);
        return Err(error.into());
    }
    drop(checksum_file);
    if let Err(error) = fs::rename(&temporary, &destination) {
        let _ = fs::remove_file(&checksum_path);
        let _ = fs::remove_dir_all(&temporary);
        return Err(error.into());
    }
    Ok(ResolvedDependency {
        import_name: name.clone(),
        package_name: name,
        version,
        root: destination.clone(),
        manifest: destination.join(MANIFEST_FILE),
        source: format!("registry+{}", registry.display()),
        checksum: Some(checksum),
    })
}

type Locked = LockedPackage;

/// Resolves every dependency reachable from `manifest_path`, materializes
/// registry packages in the Severian cache, verifies their SHA-256 digest, and
/// writes exact selections to `sev.lock`.
pub fn resolve_dependencies(manifest_path: &Path) -> Result<Resolution, PackageError> {
    resolve_dependencies_with_lock(manifest_path, true, true)
}

/// Resolves for a dependency compilation without creating a lockfile inside
/// that dependency's source tree. The root package owns `sev.lock`.
pub fn resolve_dependencies_transient(manifest_path: &Path) -> Result<Resolution, PackageError> {
    resolve_dependencies_with_lock(manifest_path, true, false)
}

/// Resolves against the newest compatible registry versions instead of
/// preferring existing lock selections. This is the operation behind
/// `sev update`; ordinary builds remain reproducible.
pub fn update_dependencies(manifest_path: &Path) -> Result<Resolution, PackageError> {
    resolve_dependencies_with_lock(manifest_path, false, true)
}

fn resolve_dependencies_with_lock(
    manifest_path: &Path,
    prefer_lock: bool,
    write_lock: bool,
) -> Result<Resolution, PackageError> {
    let manifest_path = manifest_path.canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "package manifest {} is invalid: {error}",
            manifest_path.display()
        ))
    })?;
    let project_root = manifest_path
        .parent()
        .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
    let lockfile = project_root.join(LOCK_FILE);
    let locked = if prefer_lock {
        read_lockfile(&lockfile)?
    } else {
        BTreeMap::new()
    };
    let mut state = Resolver {
        locked,
        resolving: HashSet::new(),
        resolved: BTreeMap::new(),
        order: Vec::new(),
    };
    state.visit_manifest(&manifest_path)?;
    let dependencies = state
        .order
        .into_iter()
        .filter_map(|name| state.resolved.remove(&name))
        .collect::<Vec<_>>();
    if write_lock {
        preserve_external_lock(&lockfile, lockfile::packages_from_resolution(&dependencies))?;
    }
    Ok(Resolution {
        dependencies,
        lockfile,
    })
}

struct Resolver {
    locked: BTreeMap<(String, String), Locked>,
    resolving: HashSet<PathBuf>,
    resolved: BTreeMap<String, ResolvedDependency>,
    order: Vec<String>,
}

impl Resolver {
    fn visit_manifest(&mut self, manifest_path: &Path) -> Result<(), PackageError> {
        let canonical = manifest_path.canonicalize().map_err(|error| {
            PackageError::Manifest(format!(
                "dependency manifest {} is invalid: {error}",
                manifest_path.display()
            ))
        })?;
        if !self.resolving.insert(canonical.clone()) {
            return Err(PackageError::Manifest(format!(
                "dependency cycle includes {}",
                canonical.display()
            )));
        }
        let manifest = parse_manifest(&canonical)?;
        let root = canonical
            .parent()
            .ok_or_else(|| PackageError::Manifest("manifest has no parent directory".into()))?;
        validate_package_payload(root)?;
        if let Some(dependencies) = manifest.get("dependencies").and_then(toml::Value::as_table) {
            for (import_name, specification) in dependencies {
                let dependency = self.resolve_one(import_name, specification, root)?;
                if let Some(existing) = self.resolved.get(import_name) {
                    if existing.package_name != dependency.package_name
                        || existing.version != dependency.version
                        || existing.root != dependency.root
                    {
                        return Err(PackageError::Manifest(format!(
                            "import `{import_name}` resolves to both {} {} and {} {}",
                            existing.package_name,
                            existing.version,
                            dependency.package_name,
                            dependency.version
                        )));
                    }
                    continue;
                }
                self.visit_manifest(&dependency.manifest)?;
                self.resolved.insert(import_name.clone(), dependency);
                self.order.push(import_name.clone());
            }
        }
        self.resolving.remove(&canonical);
        Ok(())
    }

    fn resolve_one(
        &self,
        import_name: &str,
        specification: &toml::Value,
        parent: &Path,
    ) -> Result<ResolvedDependency, PackageError> {
        validate_import_name(import_name)?;
        let detail = specification.as_table();
        let package = detail
            .and_then(|table| table.get("package"))
            .and_then(toml::Value::as_str)
            .unwrap_or(import_name);
        validate_package_name(package)?;
        if let Some(path) = detail
            .and_then(|table| table.get("path"))
            .and_then(toml::Value::as_str)
        {
            return resolve_path(import_name, package, parent, path, specification);
        }
        let requirement = specification.as_str().or_else(|| {
            detail
                .and_then(|table| table.get("version"))
                .and_then(toml::Value::as_str)
        });
        let requirement = requirement.ok_or_else(|| {
            PackageError::Manifest(format!(
                "dependency `{import_name}` must specify a version or path"
            ))
        })?;
        let explicit_registry = detail
            .and_then(|table| table.get("registry"))
            .and_then(toml::Value::as_str)
            .map(PathBuf::from);
        let mut registry = explicit_registry
            .clone()
            .or_else(|| std::env::var_os("SEVERIAN_REGISTRY").map(PathBuf::from))
            .unwrap_or_else(default_registry_root);
        if explicit_registry.is_some()
            && !registry.is_absolute()
            && !registry.to_string_lossy().starts_with("file://")
        {
            registry = parent.join(registry);
        }
        let registry = strip_file_scheme(&registry)?;
        let source = format!("registry+{}", registry.display());
        let locked = self.locked.get(&(package.to_owned(), source.clone()));
        resolve_registry(import_name, package, requirement, &registry, locked)
    }
}

fn resolve_path(
    import_name: &str,
    package: &str,
    parent: &Path,
    relative: &str,
    specification: &toml::Value,
) -> Result<ResolvedDependency, PackageError> {
    let root = parent.join(relative).canonicalize().map_err(|error| {
        PackageError::Manifest(format!(
            "dependency `{import_name}` has invalid path {}: {error}",
            parent.join(relative).display()
        ))
    })?;
    let manifest = manifest_in(&root).unwrap_or_else(|| root.join(MANIFEST_FILE));
    let value = parse_manifest(&manifest)?;
    let declared = package_name(&value, &manifest)?;
    if declared != package {
        return Err(PackageError::Manifest(format!(
            "dependency `{import_name}` expects package `{package}` but resolves to `{declared}`"
        )));
    }
    let version = package_version(&value, &manifest)?;
    if let Some(requirement) = specification
        .as_table()
        .and_then(|table| table.get("version"))
        .and_then(toml::Value::as_str)
    {
        if !version_matches(requirement, &version) {
            return Err(PackageError::Manifest(format!(
                "path dependency `{import_name}` is version {version}, which does not satisfy `{requirement}`"
            )));
        }
    }
    Ok(ResolvedDependency {
        import_name: import_name.into(),
        package_name: declared.into(),
        version,
        root,
        manifest,
        source: format!("path+{}", parent.join(relative).display()),
        checksum: None,
    })
}

fn resolve_registry(
    import_name: &str,
    package: &str,
    requirement: &str,
    registry: &Path,
    locked: Option<&Locked>,
) -> Result<ResolvedDependency, PackageError> {
    let package_root = registry.join("packages").join(package);
    let mut versions = fs::read_dir(&package_root)
        .map_err(|error| {
            PackageError::Manifest(format!(
                "could not query package `{package}` in registry {}: {error}",
                registry.display()
            ))
        })?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|version| parse_version(version).is_some() && version_matches(requirement, version))
        .collect::<Vec<_>>();
    versions.sort_by(|left, right| compare_versions(left, right));
    let selected = locked
        .filter(|entry| version_matches(requirement, &entry.version))
        .filter(|entry| versions.iter().any(|version| version == &entry.version))
        .map(|entry| entry.version.clone())
        .or_else(|| versions.pop())
        .ok_or_else(|| {
            PackageError::Manifest(format!(
                "registry {} has no version of `{package}` matching `{requirement}`",
                registry.display()
            ))
        })?;
    let source_root = package_root.join(&selected);
    let expected = read_registry_checksum(registry, package, &selected)?;
    let actual = directory_checksum(&source_root)?;
    if expected != actual {
        return Err(PackageError::Manifest(format!(
            "checksum mismatch for `{package}` {selected}: expected {expected}, got {actual}"
        )));
    }
    if let Some(locked) = locked {
        if locked
            .checksum
            .as_deref()
            .is_some_and(|checksum| checksum != actual)
            || locked.source != format!("registry+{}", registry.display())
        {
            return Err(PackageError::Manifest(format!(
                "locked checksum or source for `{package}` {selected} no longer matches the registry"
            )));
        }
    }
    let cache_root = package_cache_root().join(package).join(&selected);
    materialize_cache(&source_root, &cache_root, &actual)?;
    let manifest = cache_root.join(MANIFEST_FILE);
    let value = parse_manifest(&manifest)?;
    let declared = package_name(&value, &manifest)?;
    let declared_version = package_version(&value, &manifest)?;
    if declared != package || declared_version != selected {
        return Err(PackageError::Manifest(format!(
            "registry entry `{package}` {selected} contains package `{declared}` {declared_version}"
        )));
    }
    Ok(ResolvedDependency {
        import_name: import_name.into(),
        package_name: package.into(),
        version: selected,
        root: cache_root,
        manifest,
        source: format!("registry+{}", registry.display()),
        checksum: Some(actual),
    })
}

fn validate_package_name(name: &str) -> Result<(), PackageError> {
    if name.is_empty()
        || name.split('.').any(str::is_empty)
        || !name
            .bytes()
            .all(|byte| matches!(byte, b'-' | b'_' | b'.') || byte.is_ascii_alphanumeric())
    {
        return Err(PackageError::Manifest(format!(
            "invalid package name `{name}`; use dot-separated ASCII letters, numbers, `-`, or `_`"
        )));
    }
    Ok(())
}

fn validate_import_name(name: &str) -> Result<(), PackageError> {
    let valid = !name.is_empty()
        && name.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            bytes
                .next()
                .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
                && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        });
    if !valid {
        return Err(PackageError::Manifest(format!(
            "invalid import name `{name}`; dependency keys must be Severian module names"
        )));
    }
    Ok(())
}

fn default_severian_home() -> PathBuf {
    std::env::var_os("SEVERIAN_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".sev")))
        .unwrap_or_else(|| PathBuf::from(".sev"))
}

fn default_registry_root() -> PathBuf {
    default_severian_home().join("registry")
}

fn package_cache_root() -> PathBuf {
    default_severian_home().join("packages")
}

fn strip_file_scheme(path: &Path) -> Result<PathBuf, PackageError> {
    let text = path.to_string_lossy();
    if let Some(path) = text.strip_prefix("file://") {
        return Ok(PathBuf::from(path));
    }
    if text.contains("://") {
        return Err(PackageError::Manifest(format!(
            "registry `{text}` is not supported by this build; configure an on-disk or file:// registry"
        )));
    }
    Ok(path.to_path_buf())
}

fn read_registry_checksum(
    registry: &Path,
    package: &str,
    version: &str,
) -> Result<String, PackageError> {
    let path = registry
        .join("checksums")
        .join(package)
        .join(format!("{version}.sha256"));
    let checksum = fs::read_to_string(&path).map_err(|error| {
        PackageError::Manifest(format!(
            "registry package `{package}` {version} has no trusted checksum at {}: {error}",
            path.display()
        ))
    })?;
    let checksum = checksum.trim().to_ascii_lowercase();
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PackageError::Manifest(format!(
            "invalid SHA-256 checksum in {}",
            path.display()
        )));
    }
    Ok(checksum)
}

fn directory_checksum(root: &Path) -> Result<String, PackageError> {
    let mut files = Vec::new();
    collect_package_files(root, root, &mut files)?;
    files.sort();
    let mut digest = Sha256::new();
    for relative in files {
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        digest.update(fs::read(root.join(&relative))?);
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn collect_package_files(
    root: &Path,
    directory: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), PackageError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root).map_err(|_| {
            PackageError::Manifest("registry package escaped its source root".into())
        })?;
        if relative
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err(PackageError::Manifest(
                "registry package contains an unsafe path".into(),
            ));
        }
        if entry.file_type()?.is_symlink() {
            return Err(PackageError::Manifest(format!(
                "registry package contains unsupported symbolic link {}",
                relative.display()
            )));
        }
        if path.is_dir() {
            if generated_package_path(relative) {
                continue;
            }
            collect_package_files(root, &path, output)?;
        } else if path.is_file() && !generated_package_path(relative) {
            output.push(relative.to_path_buf());
        }
    }
    Ok(())
}

fn materialize_cache(
    source: &Path,
    destination: &Path,
    checksum: &str,
) -> Result<(), PackageError> {
    let marker = destination.join(".checksum");
    if fs::read_to_string(&marker).is_ok_and(|value| value.trim() == checksum)
        && directory_checksum(destination).is_ok_and(|actual| actual == checksum)
    {
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| PackageError::Manifest("package cache path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("package"),
        std::process::id()
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    copy_tree(source, &temporary)?;
    fs::write(temporary.join(".checksum"), format!("{checksum}\n"))?;
    if destination.exists() {
        fs::remove_dir_all(destination)?;
    }
    fs::rename(temporary, destination)?;
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), PackageError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(source).unwrap_or(&path);
        if matches!(
            entry.file_name().to_str(),
            Some("target" | ".git" | "sev.lock" | ".checksum")
        ) {
            continue;
        }
        if entry.file_type()?.is_symlink() {
            return Err(PackageError::Manifest(format!(
                "package contains unsupported symbolic link {}",
                relative.display()
            )));
        }
        let output = destination.join(entry.file_name());
        if path.is_dir() {
            copy_tree(&path, &output)?;
        } else if path.is_file() {
            fs::copy(path, output)?;
        }
    }
    Ok(())
}

fn generated_package_path(relative: &Path) -> bool {
    relative.components().any(|component| {
        matches!(
            component,
            Component::Normal(name)
                if name == "target" || name == ".git" || name == "sev.lock" || name == ".checksum"
        )
    })
}

fn read_lockfile(path: &Path) -> Result<BTreeMap<(String, String), Locked>, PackageError> {
    Ok(lockfile::read(path)?
        .packages
        .into_iter()
        .map(|package| ((package.name.clone(), package.source.clone()), package))
        .collect())
}

fn package_version(manifest: &toml::Value, path: &Path) -> Result<String, PackageError> {
    manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|package| package.get("version"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PackageError::Manifest(format!("{} is missing `package.version`", path.display()))
        })
}

fn parse_version(value: &str) -> Option<Vec<u64>> {
    let value = value.split_once('-').map_or(value, |(core, _)| core);
    let parts = value
        .split('.')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!parts.is_empty() && parts.len() <= 3).then_some(parts)
}

fn normalized_version(value: &str) -> Option<[u64; 3]> {
    let parts = parse_version(value)?;
    Some([
        parts.first().copied().unwrap_or(0),
        parts.get(1).copied().unwrap_or(0),
        parts.get(2).copied().unwrap_or(0),
    ])
}

fn compare_versions(left: &String, right: &String) -> std::cmp::Ordering {
    normalized_version(left).cmp(&normalized_version(right))
}

fn version_matches(requirement: &str, version: &str) -> bool {
    let Some(candidate) = normalized_version(version) else {
        return false;
    };
    let requirement = requirement.trim();
    if requirement == "*" {
        return true;
    }
    if requirement.contains(',') {
        return requirement
            .split(',')
            .all(|part| version_matches(part.trim(), version));
    }
    for operator in [">=", "<=", ">", "=", "<"] {
        if let Some(value) = requirement.strip_prefix(operator) {
            let Some(required) = normalized_version(value.trim()) else {
                return false;
            };
            return match operator {
                ">=" => candidate >= required,
                "<=" => candidate <= required,
                ">" => candidate > required,
                "<" => candidate < required,
                "=" => candidate == required,
                _ => false,
            };
        }
    }
    let (kind, raw) = if let Some(raw) = requirement.strip_prefix('~') {
        ('~', raw.trim())
    } else if let Some(raw) = requirement.strip_prefix('^') {
        ('^', raw.trim())
    } else {
        ('^', requirement)
    };
    if raw.contains('*') || raw.contains('x') || raw.contains('X') {
        let required = raw.split('.').collect::<Vec<_>>();
        let actual = version.split('.').collect::<Vec<_>>();
        return required.iter().enumerate().all(|(index, part)| {
            matches!(*part, "*" | "x" | "X") || actual.get(index).is_some_and(|value| value == part)
        });
    }
    let Some(lower) = normalized_version(raw) else {
        return false;
    };
    if candidate < lower {
        return false;
    }
    let components = parse_version(raw).map_or(0, |parts| parts.len());
    let upper = if kind == '~' {
        if components <= 1 {
            [lower[0] + 1, 0, 0]
        } else {
            [lower[0], lower[1] + 1, 0]
        }
    } else if lower[0] > 0 {
        [lower[0] + 1, 0, 0]
    } else if lower[1] > 0 {
        [0, lower[1] + 1, 0]
    } else {
        [0, 0, lower[2] + 1]
    };
    candidate < upper
}

#[cfg(test)]
mod tests {
    use super::version_matches;

    #[test]
    fn cargo_style_version_requirements_select_compatible_releases() {
        assert!(version_matches("1.4", "1.4.2"));
        assert!(version_matches("1.4", "1.9.0"));
        assert!(!version_matches("1.4", "2.0.0"));
        assert!(version_matches("^0.8", "0.8.4"));
        assert!(!version_matches("^0.8", "0.9.0"));
        assert!(version_matches("~2.1", "2.1.9"));
        assert!(!version_matches("~2.1", "2.2.0"));
        assert!(version_matches(">=1.2, <2.0", "1.8.0"));
    }
}
