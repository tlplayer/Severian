use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn source(name: &str, contents: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "severian-contract-{name}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("contract.sev");
    std::fs::write(&path, contents).unwrap();
    path
}

fn run_test(path: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("test")
        .arg(path)
        .output()
        .unwrap()
}

#[test]
fn enforces_valid_and_invalid_function_preconditions() {
    let valid = source(
        "valid-precondition",
        concat!(
            "def positive(x: int) -> int with\n",
            "{\n    x >= 0,\n}:\n",
            "    return x\n",
            "test \"valid\":\n    assert(positive(1) == 1)\n",
        ),
    );
    let output = run_test(&valid);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let invalid = source(
        "invalid-precondition",
        concat!(
            "def positive(x: int) -> int with\n",
            "{\n    x >= 0 -> exception(\"x must be positive\", location, vars),\n}:\n",
            "    return x\n",
            "test \"invalid\":\n    positive(-1)\n",
        ),
    );
    let output = run_test(&invalid);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("contract error: x must be positive"));
    assert!(stderr.contains("location:"));
    assert!(stderr.contains("vars: x"));
}

#[test]
fn rechecks_a_deferred_condition_after_its_list_dependency_changes() {
    let path = source(
        "deferred",
        concat!(
            "def change(x: list[int]) with\n",
            "{\n    defer len(x) < 3 -> exception(\"deferred failed\", location, vars),\n}:\n",
            "    x.append(3)\n",
            "test \"deferred\":\n    change([1, 2])\n",
        ),
    );
    let output = run_test(&path);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("contract error: deferred failed"));

    let test_contract = source(
        "deferred-test",
        concat!(
            "list values = [1, 2]\n",
            "def support():\n    return\n",
            "test \"deferred test contract\" with\n",
            "{\n    defer len(values) < 3 -> exception(\"test contract failed\", location, vars),\n}:\n",
            "    values.append(3)\n",
        ),
    );
    let output = run_test(&test_contract);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("contract error: test contract failed")
    );
}

#[test]
fn profile_contracts_enforce_both_bounds_and_reject_stubbed_lower_bounds() {
    let passing = source(
        "profile-pass",
        concat!(
            "def measured():\n    return\n",
            "test with profile \"bounded\" -> TestResult with\n",
            "{\n    defer 0ms < time < 2s -> exception(\"runtime outside range\"),\n}:\n",
            "    measured()\n",
        ),
    );
    let output = run_test(&passing);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stubbed = source(
        "profile-stub",
        concat!(
            "def stub():\n    return\n",
            "test with profile \"stub\" -> TestResult with\n",
            "{\n    defer 1ms < time < 2s -> exception(\"stub was not measured\", location, vars),\n}:\n",
            "    stub()\n",
        ),
    );
    let output = run_test(&stubbed);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("contract error: stub was not measured"));
    assert!(stderr.contains("vars: time"));

    let upper = source(
        "profile-upper",
        concat!(
            "def work():\n    return\n",
            "test with profile \"upper\" -> TestResult with\n",
            "{\n    defer time < 0ms -> exception(\"upper runtime bound failed\", location, vars),\n}:\n",
            "    work()\n",
        ),
    );
    let output = run_test(&upper);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("contract error: upper runtime bound failed"));
}

#[test]
fn formatter_canonicalizes_contract_layout() {
    let path = source(
        "format",
        "def value(x: int) -> int with { x >= 0, defer x < 10 -> exception(\"too   large\"), }: \n    return x\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("fmt")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let formatted = std::fs::read_to_string(path).unwrap();
    assert!(formatted.contains(" with\n{\n    x >= 0,\n    defer x < 10"));
    assert!(formatted.contains("exception(\"too   large\")"));
    assert!(formatted.contains("\n}:\n"));
}

#[test]
fn profile_flag_runs_only_profile_tests() {
    let path = source(
        "profile-selection",
        concat!(
            "def work():\n    return\n",
            "test \"ordinary failure\":\n    assert(false)\n",
            "test with profile \"profile success\" -> TestResult with\n",
            "{\n    defer time < 2s,\n}:\n",
            "    work()\n",
        ),
    );
    let profile = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("test")
        .arg("--profile")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        profile.status.success(),
        "{}",
        String::from_utf8_lossy(&profile.stderr)
    );
    assert!(String::from_utf8_lossy(&profile.stdout).contains("1 profile test(s) passed"));

    let all = run_test(&path);
    assert!(!all.status.success());
}

#[test]
fn profile_memory_mode_reports_speed_and_allocation_metrics() {
    let path = source(
        "profile-memory",
        concat!(
            "def measured() -> int:\n    return 42\n",
            "test with profile \"developer diagnostics\" -> TestResult with\n",
            "{\n",
            "    defer time < 2s,\n",
            "    defer memory < 32mb,\n",
            "    defer allocations < 10000,\n",
            "}:\n",
            "    assert(measured() == 42)\n",
        ),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("test")
        .arg("--profile")
        .arg("--memory")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Memory + profile checking"));
    assert!(stdout.contains("Profile developer diagnostics"));
    assert!(stdout.contains("time_ns "));
    assert!(stdout.contains("allocated_bytes "));
    assert!(stdout.contains("allocations "));
    assert!(stdout.contains("Memory + profile summary"));
}

#[test]
fn leak_checking_requires_address_sanitizer() {
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args([
            "test",
            "--profile",
            "--memory",
            "--leaks",
            "--sanitizer",
            "undefined",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("`--leaks` requires the address sanitizer"));
}
