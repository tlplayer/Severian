use severian_driver::{compile_native, compile_path, compile_source, run, run_tests};
use std::path::PathBuf;
use std::process::Command;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/01-values-control/03-basic-functions.sev")
}

#[test]
fn runs_functions_bindings_and_conditionals() {
    let compilation = compile_path(&fixture()).unwrap();
    let mut output = Vec::new();
    run(&compilation.hir, |line| output.push(line.to_owned())).unwrap();
    assert_eq!(output, ["large"]);
}

#[test]
fn runs_attached_severian_tests() {
    let compilation = compile_path(&fixture()).unwrap();
    let mut report = Vec::new();
    let passed = run_tests(&compilation.hir, |line| report.push(line.to_owned())).unwrap();
    assert_eq!(passed, 1);
    assert_eq!(report, ["test add ... ok"]);
}

#[test]
fn reports_a_failed_severian_assertion() {
    let compilation =
        compile_source("def value() -> int:\n    return 1\n\ntest:\n    assert(value() == 2)\n")
            .unwrap();
    let error = run_tests(&compilation.hir, |_| {}).unwrap_err();
    assert!(error
        .to_string()
        .contains("test `value` failed: execution error: assertion failed"));
}

#[test]
fn compiles_basic_functions_to_a_native_executable() {
    let compilation = compile_path(&fixture()).unwrap();
    let output_path = std::env::temp_dir().join(format!("severian-basic-{}", std::process::id()));
    compile_native(&compilation, &output_path).unwrap();

    let output = Command::new(&output_path).output().unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "large\n");
    std::fs::remove_file(output_path).unwrap();
}
