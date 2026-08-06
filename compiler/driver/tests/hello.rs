use std::path::PathBuf;
use std::process::Command;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/00-getting-started/01-hello.sev")
}

#[test]
fn checks_the_hello_fixture() {
    let status = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(fixture())
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn runs_the_hello_fixture() {
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(fixture())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "hello, severian\n"
    );
}

#[test]
fn direct_source_invocation_compiles_and_runs_native_code() {
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(fixture())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "hello, severian\n"
    );
}

#[test]
fn direct_source_invocation_executes_native_tests_when_main_is_absent() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/26-problems/06-min-cost-climbing-stairs.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(fixture)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "3 passed\n");
}
