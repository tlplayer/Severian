use crate::lockfile::{self};
use crate::manifest::read_installation_manifest;
use crate::resolver::{resolve_external_requirements, InstallPlanItem, VendorCatalog};
use crate::sandbox::validate_package_payload;
use crate::signature::verify_sha256;
use crate::system::verify_declared_tools;
use crate::transport::download_verified;
use crate::trust::{severian_home, Date, TrustRegistry};
use crate::{resolve_dependencies_transient, PackageError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallationPlan {
    pub package_name: String,
    pub manifest: PathBuf,
    pub lockfile: PathBuf,
    pub items: Vec<InstallPlanItem>,
    packages: Vec<crate::lockfile::LockedPackage>,
    pub locked: bool,
}

pub fn plan_installation(manifest: &Path, locked: bool) -> Result<InstallationPlan, PackageError> {
    let resolution = resolve_dependencies_transient(manifest)?;
    let packages = lockfile::packages_from_resolution(&resolution.dependencies);
    let root_manifest = read_installation_manifest(manifest)?;
    validate_package_payload(&root_manifest.root)?;
    verify_declared_tools(&root_manifest)?;
    let mut manifests = vec![root_manifest.clone()];
    for dependency in &resolution.dependencies {
        validate_package_payload(&dependency.root)?;
        let dependency_manifest = read_installation_manifest(&dependency.manifest)?;
        verify_declared_tools(&dependency_manifest)?;
        manifests.push(dependency_manifest);
    }
    let has_external = manifests
        .iter()
        .any(|manifest| !manifest.install.is_empty());
    let mut items = Vec::new();
    if has_external {
        let trust = TrustRegistry::load_default()?;
        let catalog = VendorCatalog::load_default()?;
        let today = Date::today();
        for package in &manifests {
            items.extend(resolve_external_requirements(
                package, &trust, &catalog, &today,
            )?);
        }
    }
    let mut unique = BTreeMap::<String, InstallPlanItem>::new();
    for item in items {
        if let Some(existing) = unique.get(&item.locked.name) {
            if existing.locked != item.locked {
                return Err(PackageError::Manifest(format!(
                    "external requirement `{}` resolves to conflicting exact versions",
                    item.locked.name
                )));
            }
        } else {
            unique.insert(item.locked.name.clone(), item);
        }
    }
    let items = unique.into_values().collect::<Vec<_>>();
    if locked {
        verify_locked_state(&resolution.lockfile, &packages, &items)?;
    }
    Ok(InstallationPlan {
        package_name: root_manifest.package_name,
        manifest: manifest.to_path_buf(),
        lockfile: resolution.lockfile,
        items,
        packages,
        locked,
    })
}

pub fn perform_installation(plan: &InstallationPlan) -> Result<(), PackageError> {
    let home = severian_home();
    for item in &plan.items {
        let cached = home.join("downloads").join(&item.locked.sha256);
        let source = if cached.is_file() {
            cached
        } else if let Some(artifact) = &item.artifact {
            artifact.clone()
        } else {
            download_verified(&item.locked.source, &item.locked.sha256, &cached)?
        };
        verify_sha256(&source, &item.locked.sha256)?;
        let destination = home
            .join("external")
            .join(&item.locked.name)
            .join(&item.locked.version);
        fs::create_dir_all(&destination)?;
        let output = destination.join("artifact");
        let temporary = destination.join("artifact.tmp");
        fs::copy(&source, &temporary)?;
        verify_sha256(&temporary, &item.locked.sha256)?;
        fs::rename(temporary, output)?;
    }
    let mut lock = lockfile::read(&plan.lockfile)?;
    lock.packages = plan.packages.clone();
    lock.external = plan.items.iter().map(|item| item.locked.clone()).collect();
    lockfile::write(&plan.lockfile, &lock)
}

pub fn verify_installation(manifest: &Path) -> Result<InstallationPlan, PackageError> {
    let plan = plan_installation(manifest, true)?;
    let home = severian_home();
    for item in &plan.items {
        let artifact = home
            .join("external")
            .join(&item.locked.name)
            .join(&item.locked.version)
            .join("artifact");
        verify_sha256(&artifact, &item.locked.sha256)?;
    }
    Ok(plan)
}

fn verify_locked_state(
    path: &Path,
    packages: &[crate::lockfile::LockedPackage],
    items: &[InstallPlanItem],
) -> Result<(), PackageError> {
    if !path.is_file() {
        return Err(PackageError::Manifest(format!(
            "{} is missing; `--locked` forbids creating it",
            path.display()
        )));
    }
    let lock = lockfile::read(path)?;
    let expected_packages = packages
        .iter()
        .map(|item| ((item.name.as_str(), item.source.as_str()), item))
        .collect::<BTreeMap<_, _>>();
    let actual_packages = lock
        .packages
        .iter()
        .map(|item| ((item.name.as_str(), item.source.as_str()), item))
        .collect::<BTreeMap<_, _>>();
    let expected = items
        .iter()
        .map(|item| (item.locked.name.as_str(), &item.locked))
        .collect::<BTreeMap<_, _>>();
    let actual = lock
        .external
        .iter()
        .map(|item| (item.name.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    if expected_packages != actual_packages
        || expected.len() != actual.len()
        || expected
            .iter()
            .any(|(name, item)| actual.get(name).is_none_or(|locked| *locked != *item))
    {
        return Err(PackageError::Manifest(format!(
            "{} does not match the exact external requirements; `--locked` forbids updating it",
            path.display()
        )));
    }
    Ok(())
}

pub fn preserve_external_lock(
    path: &Path,
    packages: Vec<crate::lockfile::LockedPackage>,
) -> Result<(), PackageError> {
    let mut lock = lockfile::read(path)?;
    lock.packages = packages;
    lockfile::write(path, &lock)
}
