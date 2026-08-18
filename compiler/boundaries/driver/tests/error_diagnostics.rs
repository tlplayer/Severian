use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn source_syntax_errors_show_file_line_and_snippet_instead_of_byte_offsets() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/syntax/E000102-inconsistent-indentation.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("E000102"));
    assert!(stderr.contains("E000102-inconsistent-indentation.sev:7:1"));
    assert!(stderr.contains("7 |   print(\"second\")"));
    assert!(!stderr.contains("at bytes"));
    assert!(!stderr.contains("package error"));
}

#[test]
fn missing_switch_colon_shows_a_machine_applicable_edit() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/syntax/E000104-missing-switch-colon.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("error[E000104]: expected `:` after match pattern"));
    assert!(stderr.contains("^ syntax needs another token here"));
    assert!(stderr.contains("help: insert `:`"));
    assert!(stderr.contains("\".yaml\" | \".yml\":"));
    assert!(stderr.contains("sev explain E000104"));
    assert!(!stderr.contains("error: package error:"));
}

#[test]
fn type_mismatch_explains_the_boundary_and_previews_conversion() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/types/E000202-incompatible-argument.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("error[E000202]: mismatched types: expected `int`, found `string`"));
    assert!(stderr
        .contains("note: this expression has type `string`, but this boundary requires `int`"));
    assert!(stderr.contains("help: convert the value with `int(...)`"));
    assert!(stderr.contains("print(double(int(\"two\")))"));
    assert!(stderr.contains("sev explain E000202"));
}

#[test]
fn missing_argument_shows_requirement_origin_and_named_argument_edit() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/types/E000203-missing-required-argument.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("error[E000203]: missing argument `device`"));
    assert!(stderr.contains("^^^^^^^^^^^^ required argument is absent"));
    assert!(stderr.contains("------ `device` is declared without a default"));
    assert!(stderr.contains("required parameter `device: string`"));
    assert!(stderr.contains("load(\"Qwen\", device = \"\")"));
    assert!(stderr.contains("sev explain E000203"));
}

#[test]
fn tensor_mismatch_reports_shapes_without_backend_vocabulary() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/types/E002401-incompatible-tensor-dimensions.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("error[E002401]: incompatible tensor dimensions"));
    assert!(stderr.contains("Tensor[f32, 32, 768]"));
    assert!(stderr.contains("Tensor[f32, 1024, 4096]"));
    assert!(stderr.contains("matrix multiplication requires `768 == 1024`"));
    assert!(stderr.contains("help: reshape, transpose, or replace an operand"));
    assert!(!stderr.contains("stablehlo.dot_general"));
    assert!(!stderr.contains("verifier error"));
}

#[test]
fn direct_function_only_source_compiles_a_linkable_module() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-function-module-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("valid_declaration.sev");
    std::fs::write(&source, "def valid_declaration() -> int:\n    return 1\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let module = root.join("target/debug/valid_declaration.ll");
    assert!(module.is_file(), "missing {}", module.display());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Compiled 1 function(s)"));
    let _ = std::fs::remove_dir_all(root);
}
