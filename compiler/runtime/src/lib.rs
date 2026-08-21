#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

/// Native runtime translation units linked by artifact backends.
///
/// Runtime owns these implementations. Backends only pass the sources to the
/// platform linker after lowering has selected versioned runtime symbols.
pub fn native_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("native");
    ["coverage.c", "string.c", "io.c"]
        .into_iter()
        .map(|source| root.join(source))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_native_runtime_translation_unit_exists() {
        for source in native_sources() {
            assert!(source.is_file(), "missing {}", source.display());
        }
    }
}
