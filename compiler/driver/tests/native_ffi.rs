use severian_driver::{
    compile_native, compile_native_integration_tests, compile_native_tests, compile_path,
    compile_source, native_integration_test_count,
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
fn c_function_runs_with_source_owned_abi_and_ffi_descriptors() {
    let (root, source) = temporary_package(
        "ffi-probe",
        std::env::consts::OS,
        "#include <stdint.h>\nint32_t sev_abi_v1_test_probe(int32_t value) { return value + 1; }\n",
    );
    let library = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library");
    std::fs::write(
        root.join("package.toml"),
        format!(
            "[package]\nname = \"ffi-probe\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"native-abi\"]\nsources = [\"src/lib.sev\"]\n\n[lib]\npath = \"src/lib.sev\"\n\n[dependencies]\nabi = {{ path = \"{}\", version = \"0.1.0\" }}\nffi = {{ path = \"{}\", version = \"0.1.0\" }}\n\n[[ffi.c]]\nname = \"provider\"\nabi = \"c-v1\"\ntargets = [\"{}\"]\nsources = [\"native/provider.c\"]\n",
            library.join("abi").display(),
            library.join("ffi").display(),
            std::env::consts::OS,
        ),
    )
    .unwrap();
    std::fs::write(
        &source,
        r#"import abi
import ffi

unsafe:
    extern("sev_abi_v1_test_probe") def probe(value: i32) -> i32

def main():
    provider = ffi.library("provider")
    probe_signature = abi.c().function(
        "sev_abi_v1_test_probe",
        [abi.Type("i32", abi.copy(), false)],
        abi.Type("i32", abi.copy(), false),
    )
    probe_symbol = provider.symbol("sev_abi_v1_test_probe", probe_signature)
    assert(probe_symbol.name == "sev_abi_v1_test_probe")
    assert(probe(41) == 42)
"#,
    )
    .unwrap();

    let compilation = compile_path(&source).unwrap();
    assert!(compilation.mir.functions.iter().any(|function| {
        function.foreign_calls.iter().any(|call| {
            call.function.symbol.as_str() == "sev_abi_v1_test_probe"
                && call.function.parameters[0].ty == severian_abi::AbiType::I32
                && call.function.result.ty == severian_abi::AbiType::I32
        })
    }));
    let executable = root.join("program");
    compile_native(&compilation, &executable).unwrap();
    let output = Command::new(&executable).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn foreign_call_arguments_and_results_are_type_checked() {
    let provider =
        "#include <stdint.h>\nint32_t sev_abi_v1_test_probe(int32_t value) { return value; }\n";
    let (argument_root, argument_source) =
        temporary_package("ffi-argument-type", std::env::consts::OS, provider);
    std::fs::write(
        &argument_source,
        "unsafe:\n    extern(\"sev_abi_v1_test_probe\") def probe(value: i32) -> i32\n\ndef main():\n    probe(\"wrong\")\n",
    )
    .unwrap();
    let argument_error = compile_path(&argument_source).unwrap_err().to_string();
    assert!(argument_error.contains("argument"));
    assert!(argument_error.contains("i32") || argument_error.contains("int"));
    std::fs::remove_dir_all(argument_root).unwrap();

    let (result_root, result_source) =
        temporary_package("ffi-result-type", std::env::consts::OS, provider);
    std::fs::write(
        &result_source,
        "unsafe:\n    extern(\"sev_abi_v1_test_probe\") def probe(value: i32) -> i32\n\ndef main():\n    value: string = probe(1)\n",
    )
    .unwrap();
    let result_error = compile_path(&result_source).unwrap_err().to_string();
    assert!(result_error.contains("string"));
    assert!(result_error.contains("i32") || result_error.contains("int"));
    std::fs::remove_dir_all(result_root).unwrap();
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
fn package_owned_file_text_provider_runs_as_typed_foreign_calls() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library/file/src/lib.sev");
    let compilation = compile_path(&source).unwrap();
    assert!(compilation
        .native_units
        .iter()
        .any(|unit| unit.package == "file" && unit.name == "file-posix"));
    let foreign_calls = compilation
        .mir
        .functions
        .iter()
        .flat_map(|function| &function.foreign_calls)
        .collect::<Vec<_>>();
    assert!(foreign_calls.iter().any(|call| {
        call.function.symbol.as_str() == "sev_abi_v1_file_read_text"
            && call.function.symbol.library.as_deref() == Some("file")
            && call.function.parameters[0].ty == severian_abi::AbiType::StringView
    }));
    assert!(compilation
        .mlir
        .as_str()
        .contains("@__sev_ffi_shim_sev_abi_v1_file_read_text"));

    let executable =
        std::env::temp_dir().join(format!("severian-file-c-v1-tests-{}", std::process::id()));
    let count = compile_native_tests(&compilation, &executable).unwrap();
    assert!(count > 0);
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
fn package_owned_regex_engine_uses_the_generic_foreign_call_path() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../library/regex/src/lib.sev");
    let compilation = compile_path(&source).unwrap();
    assert!(compilation
        .native_units
        .iter()
        .any(|unit| unit.package == "regex" && unit.name == "regex-posix"));
    let foreign_calls = compilation
        .mir
        .functions
        .iter()
        .flat_map(|function| &function.foreign_calls)
        .collect::<Vec<_>>();
    assert!(foreign_calls.iter().any(|call| {
        call.function.symbol.as_str() == "sev_abi_v1_regex_matches"
            && call.function.parameters[0].ty == severian_abi::AbiType::StringView
            && call.function.result.ty == severian_abi::AbiType::Bool
    }));
    assert!(compilation
        .mlir
        .as_str()
        .contains("@__sev_ffi_shim_sev_abi_v1_regex_matches"));

    let executable =
        std::env::temp_dir().join(format!("severian-regex-c-v1-tests-{}", std::process::id()));
    let count = compile_native_tests(&compilation, &executable).unwrap();
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
fn a_missing_declared_symbol_reports_the_package_and_symbol() {
    let (root, source) = temporary_package(
        "missing-symbol",
        std::env::consts::OS,
        "#include <stdint.h>\nint32_t sev_abi_v1_unrelated_probe(int32_t value) { return value; }\n",
    );
    let compilation = compile_path(&source).unwrap();
    let error = compile_native(&compilation, &root.join("program")).unwrap_err();
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("E0805"));
    assert!(diagnostic.contains("missing-symbol"));
    assert!(diagnostic.contains("sev_abi_v1_test_probe"));
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
