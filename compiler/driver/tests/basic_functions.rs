use severian_driver::{compile_path, compile_source, run, run_tests};
use std::path::PathBuf;

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
fn runs_return_and_throw_chaos_events() {
    let source = concat!(
        "def read() -> Result[list[int], IOError] | None:\n",
        "    return [65]\n",
        "\n",
        "test with chaos \"read results\":\n",
        "    chaos.add(when read return None)\n",
        "    chaos.add(when read return Failure(PermissionDenied))\n",
        "    for event in chaos:\n",
        "        result = read()\n",
        "\n",
        "test with chaos \"read exceptions\":\n",
        "    chaos.add(when read throw PermissionDenied)\n",
        "    chaos.add(when read throw TimedOut)\n",
        "    for event in chaos:\n",
        "        result = read()\n",
    );
    let compilation = compile_source(source).unwrap();
    let mut report = Vec::new();

    let passed = run_tests(&compilation.hir, |line| report.push(line.to_owned())).unwrap();

    assert_eq!(passed, 2);
    assert_eq!(
        report,
        ["test read results ... ok", "test read exceptions ... ok"]
    );
}

#[test]
fn evaluates_integer_and_fractional_powers() {
    let source = concat!(
        "def square(value: float) -> float:\n",
        "    return value ** 2\n",
        "\n",
        "test \"square\":\n",
        "    assert(square(3.0) == 9.0)\n",
        "\n",
        "def squareRoot(value: float) -> float:\n",
        "    return value ** .5\n",
        "\n",
        "test \"square root\":\n",
        "    assert(squareRoot(9.0) == 3.0)\n",
        "\n",
        "def powerTower() -> int:\n",
        "    return 2 ** 3 ** 2\n",
        "\n",
        "test \"right associative power\":\n",
        "    assert(powerTower() == 512)\n",
        "\n",
        "def negativeSquare() -> int:\n",
        "    return -2 ** 2\n",
        "\n",
        "test \"power binds before negation\":\n",
        "    assert(negativeSquare() == -4)\n",
    );
    let compilation = compile_source(source).unwrap();

    assert_eq!(run_tests(&compilation.hir, |_| {}).unwrap(), 4);
}
