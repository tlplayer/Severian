use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn source_syntax_errors_show_file_line_and_snippet_instead_of_byte_offsets() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/error/syntax/E0102-inconsistent-indentation.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&source)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("E0102"));
    assert!(stderr.contains("E0102-inconsistent-indentation.sev:7:1"));
    assert!(stderr.contains("7 |   print(\"second\")"));
    assert!(!stderr.contains("at bytes"));
    assert!(!stderr.contains("package error"));
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
