use severian_driver::{
    compile_native, compile_native_integration_tests, compile_path, compile_source,
    native_integration_test_count,
};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{path::PathBuf, process::Command};

fn temporary_package(name: &str, targets: &str, provider: &str) -> (PathBuf, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-native-ffi-{name}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("native")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"native-abi\"]\nsources = [\"src/lib.sev\"]\n\n[lib]\npath = \"src/lib.sev\"\n\n[[ffi.c]]\nname = \"provider\"\nabi = \"c-v1\"\ntargets = [\"{targets}\"]\nsources = [\"native/provider.c\"]\n"
        ),
    )
    .unwrap();
    let source = root.join("src/lib.sev");
    std::fs::write(
        &source,
        "unsafe:\n    extern(\"sev_abi_v1_test_probe\") def probe(value: i32) -> i32\n\ndef main():\n    return\n",
    )
    .unwrap();
    std::fs::write(root.join("native/provider.c"), provider).unwrap();
    (root, source)
}

#[test]
fn package_owned_network_provider_runs_through_c_v1_shims() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library/network/src/lib.sev");
    let compilation = compile_path(&source).unwrap();
    assert_eq!(native_integration_test_count(&compilation.hir), 3);
    assert!(compilation
        .native_units
        .iter()
        .any(|unit| unit.package == "network" && unit.name == "network-posix"));
    assert!(compilation
        .optimized_hir
        .metadata
        .external_functions
        .contains_key("sev_abi_v1_network_connect"));

    let executable = std::env::temp_dir().join(format!(
        "severian-network-c-v1-tests-{}",
        std::process::id()
    ));
    let count = compile_native_integration_tests(&compilation, &executable).unwrap();
    assert_eq!(count, 3);
    let output = Command::new(&executable).output().unwrap();
    if output.status.success() {
        let _ = std::fs::remove_file(&executable);
    }
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn a_missing_target_provider_is_a_severian_diagnostic() {
    let (root, source) = temporary_package(
        "missing-provider",
        "unsupported-target",
        "#include <stdint.h>\nint32_t sev_abi_v1_test_probe(int32_t value) { return value; }\n",
    );
    let compilation = compile_path(&source).unwrap();
    let error = compile_native(&compilation, &root.join("program")).unwrap_err();
    assert!(error.to_string().contains("E0804"));
    assert!(error.to_string().contains("has no c-v1 native provider"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_mismatched_c_signature_fails_during_provider_compilation() {
    let (root, source) = temporary_package(
        "signature-mismatch",
        std::env::consts::OS,
        "#include <stdint.h>\nint64_t sev_abi_v1_test_probe(int64_t value) { return value; }\n",
    );
    let compilation = compile_path(&source).unwrap();
    let error = compile_native(&compilation, &root.join("program")).unwrap_err();
    assert!(error.to_string().contains("E0806"));
    assert!(error.to_string().contains("failed ABI compilation"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn programs_without_native_imports_have_no_native_link_plan() {
    let compilation = compile_source("def main():\n    return\n").unwrap();
    assert!(compilation.native_units.is_empty());
    assert!(compilation.native_assets.is_empty());
    assert!(compilation
        .optimized_hir
        .metadata
        .external_functions
        .is_empty());
}
