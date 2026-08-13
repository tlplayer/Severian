use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-build-diagnostics-{}-{nonce}",
        std::process::id()
    ))
}

fn fixture() -> PathBuf {
    let root = temporary_directory();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("type_error.sev"),
        "def consume(value: int) -> int:\n    return value\n\ndef main():\n    print(consume(\"wrong\"))\n",
    )
    .unwrap();
    std::fs::write(
        root.join("bounds_error.sev"),
        "def main():\n    values = [1, 2, 3]\n    print(values[3])\n",
    )
    .unwrap();
    std::fs::write(
        root.join("warning.sev"),
        "def main():\n    unused = 42\n    print(\"done\")\n",
    )
    .unwrap();
    root
}

#[test]
fn build_reports_independent_files_in_one_pass() {
    let root = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["build"])
        .arg(&root)
        .args(["--max-errors", "10"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("E0202"));
    assert!(stderr.contains("E0401"));
    assert!(stderr.contains("W001"));
    assert!(stderr.contains("2 independent error(s)"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn build_json_is_bounded_and_machine_readable() {
    let root = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["build"])
        .arg(&root)
        .args(["--max-errors", "1", "--message-format", "json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with('['));
    assert!(stdout.trim_end().ends_with(']'));
    assert!(stdout.contains("\"severity\":\"error\""));
    assert_eq!(stdout.matches("\"severity\":\"error\"").count(), 1);
    assert!(stdout.contains("\"path\":"));
    assert!(stdout.contains("\"message\":"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn build_compiles_a_library_only_package_without_an_entry_point() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"function-library\"\nversion = \"0.1.0\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/lib.sev"),
        "def valid_declaration() -> int:\n    return 1\n\ntest \"valid declaration\":\n    assert(valid_declaration() == 1)\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .arg(&root)
        .arg("--verify-each")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(root.join("target/debug/lib.ll").is_file());
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("resolved HIR, linked HIR, every HIR transformation, and MIR"));
    let _ = std::fs::remove_dir_all(root);
}
