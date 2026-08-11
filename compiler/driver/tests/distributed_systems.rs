use std::path::PathBuf;
use std::process::Command;

#[test]
fn distributed_systems_labs_compile_and_execute_natively() {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runner = repository_root.join("docs/lab/distributed_systems/run_labs.sh");
    let output = Command::new(runner)
        .env("SEVERIAN_COMPILER", env!("CARGO_BIN_EXE_sev"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("All 9 distributed-systems labs passed.")
    );
}
