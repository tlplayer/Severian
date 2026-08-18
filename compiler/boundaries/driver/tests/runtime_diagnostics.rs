use std::{
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn run_source(label: &str, source: &str, arguments: &[&str]) -> (PathBuf, Output) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-runtime-diagnostic-{label}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("{label}.sev"));
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(arguments)
        .arg(&path)
        .output()
        .unwrap();
    (root, output)
}

#[test]
fn runtime_failure_has_an_e_code_source_label_and_actionable_detail() {
    let (root, output) = run_source(
        "bounds",
        "def select(index: int) -> int:\n    values = [10, 20]\n    return values[index]\n\ndef main():\n    print(select(4))\n",
        &[],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[E000910]: index is out of bounds"));
    assert!(stderr.contains("bounds.sev:3:12"));
    assert!(stderr.contains("3 |     return values[index]"));
    assert!(stderr.contains("collection index 4 is invalid; length is 2"));
    assert!(stderr.contains("sev explain E000910"));
    assert!(!stderr.contains("exited with signal"));
    assert!(!stderr.contains("error: error["));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn internal_runtime_diagnostics_include_protocol_and_artifact_context() {
    let (root, output) = run_source(
        "division",
        "def divide(value: int, divisor: int) -> int:\n    return value / divisor\n\ndef main():\n    print(divide(10, 0))\n",
        &["run", "--diagnostics=internal"],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[E000920]: division by zero"));
    assert!(
        stderr.contains("runtime-protocol=v2"),
        "unexpected diagnostic:\n{stderr}"
    );
    assert!(stderr.contains("artifact="));
    assert!(stderr.contains("source="));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn an_unmigrated_native_abort_is_reported_as_e000990() {
    let (root, output) = run_source(
        "fallback",
        "def main():\n    values = [10]\n    values.remove(2)\n",
        &[],
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(
        "error[E000990]: native program terminated without a Severian runtime diagnostic"
    ));
    assert!(stderr.contains("stack trace:"));
    assert!(stderr.contains("__sev_collection_remove"));
    assert!(!stderr.contains("rerun with `--diagnostics=internal`"));
    assert!(!stderr.contains("target/debug/fallback exited with signal"));
    std::fs::remove_dir_all(root).unwrap();
}
