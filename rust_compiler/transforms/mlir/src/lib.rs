#![deny(unsafe_op_in_unsafe_fn)]

mod emit;
mod ffi;
mod library;
pub mod structured;
mod verify;

pub use emit::{render, MlirArtifact, MlirError};
pub use library::{registered_libraries, MlirLibrary};
pub use severian_lir::{
    LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape,
    LoweredType,
};
pub use verify::{
    compose, compose_gpu_launchers, verify_artifact, GpuLaunchArtifact, VerifiedMlirArtifact,
};

/// Canonical MLIR spelling for a lowered Severian type. Custom compilers use
/// the same scalar/tensor mapping as ordinary lowering.
pub fn type_spelling(ty: &LoweredType) -> Result<String, MlirError> {
    emit::mlir_type(ty)
}

#[cfg(test)]
mod boundary_tests {
    use std::fs;
    use std::path::Path;

    fn source_files(root: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                source_files(&path, files);
            } else if matches!(
                path.extension().and_then(|extension| extension.to_str()),
                Some("rs" | "sev")
            ) {
                files.push(path);
            }
        }
    }

    #[test]
    fn source_owned_mlir_has_no_helper_name_escape_hatch() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        assert!(
            !repository
                .join("rust_compiler/transforms/mlir/src/native.rs")
                .exists(),
            "the deleted Rust OperationState interpreter must not return"
        );
        let rust_compiler = repository.join("rust_compiler");
        let mut files = Vec::new();
        source_files(&rust_compiler, &mut files);
        let forbidden_rust = [
            concat!("__sev_", "mlir_"),
            concat!("struct Operation", "State"),
            concat!("mlirOperation", "State"),
        ];
        for path in &files {
            let source = fs::read_to_string(path).unwrap();
            for forbidden in forbidden_rust {
                assert!(
                    !source.contains(forbidden),
                    "Rust compiler source contains deleted MLIR builder boundary `{forbidden}`: {}",
                    path.display()
                );
            }
        }

        files.clear();
        source_files(&repository.join("sev_compiler"), &mut files);
        let forbidden = [
            concat!("__sev_", "mlir_", "operation_draft"),
            concat!("__sev_", "mlir_", "type_draft"),
            concat!("__sev_", "mlir_", "array_draft"),
            concat!("__sev_", "mlir_", "affine_map_draft"),
        ];
        for path in &files {
            let source = fs::read_to_string(path).unwrap();
            for symbol in forbidden {
                assert!(
                    !source.contains(symbol),
                    "legacy MLIR draft surface returned in {}",
                    path.display()
                );
            }
        }
    }
}
