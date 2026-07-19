use severian_driver::{compile_path, compile_source, run_integration_tests, run_tests};
use std::path::PathBuf;

fn integration_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/15-tests/06-integration.sev")
}

#[test]
fn skips_integration_tests_by_default_and_runs_them_explicitly() {
    let compilation = compile_path(&integration_fixture()).unwrap();

    assert_eq!(run_tests(&compilation.hir, |_| {}).unwrap(), 0);

    let mut report = Vec::new();
    assert_eq!(
        run_integration_tests(&compilation, |line| report.push(line.to_owned())).unwrap(),
        1
    );
    assert_eq!(
        report,
        ["test with integration captures native output ... ok"]
    );
}

#[test]
fn reports_a_native_stdout_assertion_failure() {
    let compilation = compile_source(concat!(
        "def main():\n",
        "    print(\"actual\")\n",
        "\n",
        "test with integration \"output mismatch\":\n",
        "    main()\n",
        "    assert(\"expected\" in stdout)\n",
    ))
    .unwrap();

    let error = run_integration_tests(&compilation, |_| {}).unwrap_err();
    assert!(error
        .to_string()
        .contains("integration test `output mismatch` failed: execution error: assertion failed"));
}

#[test]
fn keeps_stdout_out_of_controlled_unit_tests() {
    let error = compile_source(concat!(
        "def main():\n",
        "    print(\"actual\")\n",
        "\n",
        "test \"not integration\":\n",
        "    assert(\"actual\" in stdout)\n",
    ))
    .unwrap_err();

    assert!(error.to_string().contains("unknown binding `stdout`"));
}
