use crate::PackageError;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildSandbox {
    pub network: bool,
    pub process_spawning: bool,
    pub package_filesystem_only: bool,
}

impl Default for BuildSandbox {
    fn default() -> Self {
        Self {
            network: false,
            process_spawning: false,
            package_filesystem_only: true,
        }
    }
}

/// Rejects executable installer payloads before package source is cached or built.
/// Severian currently has no build-script escape hatch; future code generation
/// must execute under `BuildSandbox::default()` rather than inheriting trust from
/// the package publisher.
pub fn validate_package_payload(root: &Path) -> Result<(), PackageError> {
    visit(root, root)
}

fn visit(root: &Path, directory: &Path) -> Result<(), PackageError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if entry.file_type()?.is_symlink() {
            return Err(PackageError::Manifest(format!(
                "package contains unsupported symbolic link {}",
                relative.display()
            )));
        }
        if path.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                ".git" | "target" | "node_modules" | ".codex" | ".agents"
            ) || name == ".venv"
                || name.starts_with(".venv-")
            {
                continue;
            }
            visit(root, &path)?;
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let script = matches!(extension.as_str(), "sh" | "py" | "ps1" | "bat" | "cmd");
        let installer_name = stem == "setup"
            || stem == "install"
            || stem == "bootstrap"
            || stem.starts_with("install-")
            || stem.starts_with("install_");
        // PowerShell and command files have no role in the declarative package
        // model. Shell/Python files remain valid package assets unless their
        // names identify an installer hook.
        let forbidden = name == "setup.py"
            || matches!(extension.as_str(), "ps1" | "bat" | "cmd")
            || (script && installer_name);
        if forbidden {
            return Err(PackageError::Manifest(format!(
                "package contains forbidden installer `{}`; declare requirements in package.toml",
                relative.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_local_environment_and_build_directories() {
        let root = std::env::temp_dir().join(format!(
            "severian-package-sandbox-environments-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".venv-local/bin")).unwrap();
        fs::create_dir_all(root.join("target/generated")).unwrap();
        fs::write(root.join(".venv-local/bin/Activate.ps1"), "activate").unwrap();
        fs::write(root.join("target/generated/install.sh"), "install").unwrap();

        validate_package_payload(&root).unwrap();
        let _ = fs::remove_dir_all(root);
    }
}
