#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

pub mod gpu;

/// Native runtime translation units linked by artifact backends.
///
/// Runtime owns these implementations. Backends only pass the sources to the
/// platform linker after lowering has selected versioned runtime symbols.
pub fn native_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("native");
    [
        "coverage.c",
        "string.c",
        "any.c",
        "list.c",
        "tensor.c",
        "channel.c",
        "io.c",
        "filesystem.c",
        "system.c",
    ]
    .into_iter()
    .map(|source| root.join(source))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn every_native_runtime_translation_unit_exists() {
        for source in native_sources() {
            assert!(source.is_file(), "missing {}", source.display());
        }
    }

    #[test]
    fn every_tensor_dtype_executes_with_exact_128_bit_storage() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = manifest.join("tests/tensor_dtype.c");
        let executable =
            std::env::temp_dir().join(format!("severian-tensor-dtype-{}", std::process::id()));
        let compiler = std::env::var_os("CC").unwrap_or_else(|| {
            if Path::new("/usr/bin/clang-21").is_file() {
                "/usr/bin/clang-21".into()
            } else {
                "clang".into()
            }
        });
        let compiled = Command::new(compiler)
            .arg("-std=gnu17")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg(&source)
            .arg("-lm")
            .arg("-o")
            .arg(&executable)
            .status()
            .expect("the C compiler must run");
        assert!(compiled.success(), "failed to compile {}", source.display());
        let executed = Command::new(&executable)
            .status()
            .expect("the tensor dtype runtime test must run");
        let _ = std::fs::remove_file(&executable);
        assert!(executed.success(), "tensor dtype runtime test failed");
    }
}
