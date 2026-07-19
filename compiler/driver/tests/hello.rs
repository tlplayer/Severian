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
fn compiles_the_hello_fixture_to_a_native_executable() {
    let output_path = std::env::temp_dir().join(format!("severian-hello-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("compile")
        .arg(fixture())
        .arg("-o")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let native = Command::new(&output_path).output().unwrap();
    assert!(native.status.success());
    assert_eq!(
        String::from_utf8(native.stdout).unwrap(),
        "hello, severian\n"
    );
    std::fs::remove_file(output_path).unwrap();
}
