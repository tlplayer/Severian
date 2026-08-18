use crate::manifest::InstallationManifest;
use crate::resolver::version_matches;
use crate::PackageError;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolves tool requirements that do not have a vendor installation record.
/// The executable and arguments are selected by Severian, never by package data.
pub fn verify_declared_tools(manifest: &InstallationManifest) -> Result<(), PackageError> {
    for requirement in &manifest.system {
        if manifest
            .install
            .iter()
            .any(|install| install.name == requirement.name)
        {
            continue;
        }
        let (executable, argument) = match requirement.name.as_str() {
            "cmake" => ("cmake", "--version"),
            "clang" => ("clang", "--version"),
            "llvm" => ("llvm-config", "--version"),
            "ninja" => ("ninja", "--version"),
            name => {
                return Err(PackageError::Manifest(format!(
                    "system requirement `{name}` has no compiler-owned resolver; add a trusted `[install.{name}]` provider or use a supported tool name"
                )))
            }
        };
        let path = find_executable(executable, &manifest.root).ok_or_else(|| {
            PackageError::Manifest(format!(
                "system requirement `{}` {} is not installed",
                requirement.name, requirement.version
            ))
        })?;
        let output = Command::new(&path)
            .arg(argument)
            .output()
            .map_err(|error| {
                PackageError::Manifest(format!(
                    "could not inspect system tool {}: {error}",
                    path.display()
                ))
            })?;
        if !output.status.success() {
            return Err(PackageError::Manifest(format!(
                "system tool {} failed version inspection",
                path.display()
            )));
        }
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let version = first_version(&text).ok_or_else(|| {
            PackageError::Manifest(format!(
                "could not determine the version of `{}`",
                requirement.name
            ))
        })?;
        if !version_matches(&requirement.version, &version) {
            return Err(PackageError::Manifest(format!(
                "system tool `{}` is version {version}, which does not satisfy `{}`",
                requirement.name, requirement.version
            )));
        }
    }
    Ok(())
}

fn find_executable(name: &str, package_root: &Path) -> Option<PathBuf> {
    env::split_paths(&env::var_os("PATH")?)
        .filter(|directory| directory.is_absolute() && !directory.starts_with(package_root))
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

fn first_version(text: &str) -> Option<String> {
    text.split(|character: char| !(character.is_ascii_digit() || character == '.'))
        .find(|part| {
            !part.is_empty()
                && part.contains('.')
                && part.split('.').all(|component| !component.is_empty())
        })
        .map(|part| part.trim_matches('.').to_owned())
}

#[cfg(test)]
mod tests {
    use super::first_version;

    #[test]
    fn extracts_tool_versions_without_accepting_commands_from_manifests() {
        assert_eq!(first_version("cmake version 3.31.2"), Some("3.31.2".into()));
        assert_eq!(
            first_version("clang version 19.1.0 (build)"),
            Some("19.1.0".into())
        );
    }
}
