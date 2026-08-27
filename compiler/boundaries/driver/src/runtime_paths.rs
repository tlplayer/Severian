use std::env;
use std::path::{Path, PathBuf};

/// Locate the Severian runtime tree independently of the checkout used to
/// compile the driver. Installed distributions set `SEVERIAN_HOME`; development
/// builds are discovered by walking upward from the executable before falling
/// back to the compile-time repository.
pub(crate) fn severian_home() -> PathBuf {
    if let Some(home) = env::var_os("SEVERIAN_HOME").map(PathBuf::from) {
        return home;
    }
    if let Ok(executable) = env::current_exe() {
        for ancestor in executable.ancestors().skip(1) {
            if is_runtime_home(ancestor) {
                return ancestor.to_owned();
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the driver crate is nested below the repository root")
        .to_owned()
}

pub(crate) fn library_root() -> PathBuf {
    env::var_os("SEVERIAN_LIBRARY_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = severian_home();
            let installed = home.join("share/severian/library");
            if installed.is_dir() {
                installed
            } else {
                home.join("library")
            }
        })
}

pub(crate) fn component_root() -> PathBuf {
    env::var_os("SEVERIAN_COMPONENT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = severian_home();
            let installed = home.join("lib/severian/components");
            if installed.is_dir() {
                installed
            } else {
                home.join("target/components")
            }
        })
}

fn is_runtime_home(candidate: &Path) -> bool {
    candidate
        .join("library/system/driver/components.toml")
        .is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_runtime_contains_the_standard_library() {
        assert!(library_root().join("tensor/package.toml").is_file());
    }
}
