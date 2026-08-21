use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temporary(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "severian-cli-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn sev() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sev"))
}

#[test]
fn bare_source_runs_globals_before_main() {
    let root = temporary("entry");
    let source = root.join("entry.sev");
    fs::write(
        &source,
        "print(\"global\")\nvalue := 7\ndef main():\n    observed := value\n    print(\"main\")\n",
    )
    .unwrap();
    let output = sev().arg(&source).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "global\nmain\n");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn build_emits_but_does_not_execute_and_check_emits_nothing() {
    let root = temporary("build");
    let source = root.join("hello.sev");
    let artifact = root.join("hello-bin");
    fs::write(&source, "print(\"must not run during build\")\n").unwrap();
    let checked = sev().args(["check"]).arg(&source).output().unwrap();
    assert!(checked.status.success());
    assert!(!artifact.exists());
    let built = sev()
        .args(["build"])
        .arg(&source)
        .args(["-o"])
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(built.status.success());
    assert!(artifact.exists());
    assert!(!String::from_utf8(built.stdout)
        .unwrap()
        .contains("must not run during build"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_default_run_and_dependency_initialization_are_deterministic() {
    let root = temporary("package");
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"entry\"\ndefault-run = \"server\"\n\n[[bin]]\nname = \"client\"\npath = \"src/client.sev\"\n\n[[bin]]\nname = \"server\"\npath = \"src/server.sev\"\n",
    )
    .unwrap();
    fs::write(
        root.join("src/dependency.sev"),
        "print(\"dependency\")\ndef main():\n    hidden := 9\n",
    )
    .unwrap();
    fs::write(root.join("src/client.sev"), "print(\"client\")\n").unwrap();
    fs::write(
        root.join("src/server.sev"),
        "import \"dependency.sev\" as dependency\nprint(\"server\")\ndef main():\n    print(\"main\")\n",
    )
    .unwrap();
    let output = sev().arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "dependency\nserver\nmain\n"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn config_sync_preserves_existing_values() {
    let root = temporary("sync");
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"sync\"\n\n[[bin]]\nname = \"sync\"\npath = \"src/main.sev\"\n\n[build]\nbackend = \"native\"\n",
    )
    .unwrap();
    fs::write(root.join("src/main.sev"), "print(\"sync\")\n").unwrap();
    let output = sev().args(["config", "sync"]).arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest = fs::read_to_string(root.join("package.toml")).unwrap();
    assert!(manifest.contains("backend = \"native\""));
    assert!(manifest.contains("profile = \"dev\""));
    assert!(manifest.contains("target = \"host\""));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn main_accepts_the_process_argument_value() {
    let root = temporary("arguments");
    let source = root.join("arguments.sev");
    fs::write(
        &source,
        "def main(arguments: args):\n    captured := arguments\n    print(\"arguments accepted\")\n",
    )
    .unwrap();
    let output = sev()
        .arg(&source)
        .args(["--", "first", "second"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"arguments accepted\n");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn build_emits_every_declared_binary_and_library_artifact() {
    let root = temporary("artifacts");
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"mixed\"\n\n[[bin]]\nname = \"mixed\"\npath = \"src/main.sev\"\n\n[lib]\nname = \"mixed_core\"\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    fs::write(root.join("src/main.sev"), "print(\"binary\")\n").unwrap();
    fs::write(root.join("src/lib.sev"), "library_value := 1\n").unwrap();
    let output = sev().args(["build"]).arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let target = root.join("target/host/dev");
    assert!(target.join("bin/mixed").is_file());
    let package = fs::read(target.join("pkg/mixed_core-0.1.0.pkg")).unwrap();
    assert!(package.starts_with(b"SEVPKG\0\x01"));
    assert!(package.ends_with(b"library_value := 1\n"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_library_only_package_builds_without_a_binary() {
    let root = temporary("library-only");
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"library_only\"\nversion = \"2.4.1\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    fs::write(root.join("src/lib.sev"), "library_value := 1\n").unwrap();
    let output = sev().args(["build"]).arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(root
        .join("target/host/dev/pkg/library_only-2.4.1.pkg")
        .is_file());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn config_show_reports_the_full_catalog_and_active_profile_overlay() {
    let root = temporary("show");
    fs::create_dir(root.join("src")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"show\"\n\n[[bin]]\nname = \"show\"\npath = \"src/main.sev\"\n\n[build]\nprofile = \"release\"\n\n[profile.release]\nopt-level = 2\n",
    )
    .unwrap();
    fs::write(root.join("src/main.sev"), "print(\"show\")\n").unwrap();
    let output = sev().args(["config", "show"]).arg(&root).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("language.type-safe = \"true\" # central default"));
    assert!(stdout.contains("profile.release.opt-level = \"2\" # package.toml"));
    assert!(stdout.contains("active-profile.opt-level = \"2\""));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn new_creates_a_lockfile() {
    let parent = temporary("new");
    let root = parent.join("created");
    let output = sev().args(["new"]).arg(&root).output().unwrap();
    assert!(output.status.success());
    assert!(root.join("sev.lock").is_file());
    fs::remove_dir_all(parent).unwrap();
}

#[test]
fn run_executes_a_relative_output_path_without_searching_path() {
    let root = temporary("relative-output");
    fs::write(root.join("app.sev"), "print(\"relative output\")\n").unwrap();
    let output = sev()
        .current_dir(&root)
        .args(["run", "app.sev", "-o", "app"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"relative output\n");
    assert!(root.join("app").is_file());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_discovers_named_and_anonymous_tests_and_reports_failures() {
    let root = temporary("tests");
    let passing = root.join("passing.sev");
    fs::write(
        &passing,
        "def clamp(value: int, low: int, high: int) -> int:\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n\ntest:\n    assert(clamp(4, 0, 10) == 4)\n\ntest \"named\":\n    assert(clamp(-1, 0, 10) == 0)\n    assert(clamp(12, 0, 10) == 10)\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&passing).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("test test 1 ... ok"), "{stdout}");
    assert!(stdout.contains("test named ... ok"), "{stdout}");

    let failing = root.join("failing.sev");
    fs::write(&failing, "test \"fails\":\n    assert(false)\n").unwrap();
    let output = sev().args(["test"]).arg(&failing).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains("test fails ... FAILED"));
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("failing.sev:2:5: assertion failed: false"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_runs_compiler_expectations_and_continues_after_source_errors() {
    let root = temporary("compiler-tests");
    fs::write(
        root.join("compiler.sev"),
        "def increment(value: int) -> int:\n    return value + 1\n\ntest with compiler \"type checks declarations\":\n    reject:\n        increment(\"wrong\")\n    accept:\n        value = increment(1)\n",
    )
    .unwrap();
    fs::write(
        root.join("invalid.sev"),
        "test:\n    assert([1, 2] == [1, 2])\n",
    )
    .unwrap();

    let output = sev().args(["test"]).arg(&root).output().unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("test type checks declarations ... ok"),
        "{stdout}"
    );
    assert!(
        stdout.contains("invalid.sev ... FAILED (compile)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("test result: 1 passed; 1 failed"),
        "{stdout}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_runs_benchmarks_and_captured_integration_expectations() {
    let root = temporary("test-modes");
    let source = root.join("modes.sev");
    fs::write(
        &source,
        "def main():\n    print(\"captured\")\n\ntest with bench \"bench\":\n    assert(2 + 2 == 4)\n\ntest with integ \"integration\":\n    main()\n    assert(\"captured\" in stdout)\n    assert(stderr == \"\")\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&source).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("test bench ... bench ("), "{stdout}");
    assert!(stdout.contains("test integration ... ok"), "{stdout}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_rejects_unimplemented_runner_modes_instead_of_passing_them_as_skipped() {
    let root = temporary("unsupported-test-mode");
    let source = root.join("property.sev");
    fs::write(&source, "test with property:\n    assert(true)\n").unwrap();
    let output = sev().args(["test"]).arg(&source).output().unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("FAILED (unsupported runner: property)"),
        "{stdout}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_test_invocations_do_not_replace_each_others_executables() {
    let root = temporary("concurrent-tests");
    let source = root.join("same.sev");
    fs::write(
        &source,
        "test \"first\":\n    assert(true)\n\ntest \"second\":\n    assert(true)\n",
    )
    .unwrap();
    let first = sev()
        .args(["test"])
        .arg(&source)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let second = sev()
        .args(["test"])
        .arg(&source)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    assert!(first.wait_with_output().unwrap().status.success());
    assert!(second.wait_with_output().unwrap().status.success());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn match_case_bindings_are_typed_without_magic_error_or_value_names() {
    let root = temporary("typed-match-cases");
    let source = root.join("match.sev");
    fs::write(
        &source,
        "def binding_first(result: int) -> int:\n    match result:\n        case error: int:\n            return error\n\ndef type_first(result: int) -> int:\n    match result:\n        case int failure:\n            return failure\n\ndef default_case(result: int) -> int:\n    match result:\n        case _:\n            return 9\n\ntest \"typed case bindings\":\n    assert(binding_first(4) == 4)\n    assert(type_first(5) == 5)\n    assert(default_case(6) == 9)\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&source).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_tests_include_declared_sources_and_conventional_tests_directory() {
    let root = temporary("package-test-discovery");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join("tests")).unwrap();
    fs::write(
        root.join("package.toml"),
        "[package]\nname = \"discovery\"\n\n[[bin]]\nname = \"discovery\"\npath = \"src/main.sev\"\n",
    )
    .unwrap();
    fs::write(
        root.join("src/main.sev"),
        "test \"package source\":\n    assert(true)\n",
    )
    .unwrap();
    fs::write(
        root.join("tests/basic.sev"),
        "test \"conventional test\":\n    assert(true)\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("test package source ... ok"), "{stdout}");
    assert!(stdout.contains("test conventional test ... ok"), "{stdout}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn recursive_test_discovery_does_not_run_imported_test_modules_twice() {
    let root = temporary("deduplicated-test-roots");
    fs::write(
        root.join("a.sev"),
        "import \"b.sev\" as b\n\ntest \"root test\":\n    assert(true)\n",
    )
    .unwrap();
    fs::write(
        root.join("b.sev"),
        "test \"dependency test\":\n    assert(true)\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout.matches("test dependency test ... ok").count(),
        1,
        "{stdout}"
    );
    assert_eq!(
        stdout.matches("test root test ... ok").count(),
        1,
        "{stdout}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_package_uses_linked_examples_as_independent_roots() {
    use std::os::unix::fs::symlink;

    let root = temporary("example-validation");
    let examples = root.join("canonical-examples");
    let package = root.join("validation");
    let dependency = root.join("support");
    fs::create_dir_all(&examples).unwrap();
    fs::create_dir_all(package.join("src")).unwrap();
    fs::create_dir_all(dependency.join("src")).unwrap();
    fs::write(
        dependency.join("package.toml"),
        "[package]\nname = \"support\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    fs::write(dependency.join("src/lib.sev"), "support_value := 1\n").unwrap();
    fs::write(
        examples.join("first.sev"),
        "import support\n\ndef main():\n    print(\"first\")\n\ntest \"same name\":\n    assert(true)\n",
    )
    .unwrap();
    fs::write(
        examples.join("second.sev"),
        "def main():\n    print(\"second\")\n\ntest \"same name\":\n    assert(true)\n",
    )
    .unwrap();
    fs::write(
        package.join("package.toml"),
        "[package]\nname = \"examples-validation\"\npublish = false\n\n[lib]\npath = \"src/validation.sev\"\n\n[dependencies]\nsupport = { path = \"../support\" }\n",
    )
    .unwrap();
    fs::write(package.join("src/validation.sev"), "# package anchor\n").unwrap();
    fs::write(
        package.join("validation.toml"),
        "[validation]\nsource = \"linked\"\nline-coverage = 100\nbranch-coverage = 100\n",
    )
    .unwrap();
    symlink("../canonical-examples", package.join("linked")).unwrap();

    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        stdout.matches("test same name ... ok").count(),
        2,
        "{stdout}"
    );
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("example "))
            .count(),
        2,
        "{stdout}"
    );
    assert!(
        stdout.contains("validated 2 independent source(s) and 0 package fixture(s)"),
        "{stdout}"
    );
    assert!(stdout.contains("examples-validation.json"), "{stdout}");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
fn validation_fixture(
    name: &str,
    source: &str,
    extra_validation: &str,
) -> (PathBuf, PathBuf, PathBuf) {
    use std::os::unix::fs::symlink;

    let root = temporary(name);
    let examples = root.join("examples");
    let package = root.join("validation");
    fs::create_dir_all(&examples).unwrap();
    fs::create_dir_all(package.join("src")).unwrap();
    fs::write(examples.join("example.sev"), source).unwrap();
    fs::write(
        package.join("package.toml"),
        "[package]\nname = \"examples-validation\"\npublish = false\n\n[lib]\npath = \"src/validation.sev\"\n",
    )
    .unwrap();
    fs::write(package.join("src/validation.sev"), "# anchor\n").unwrap();
    fs::write(
        package.join("validation.toml"),
        format!(
            "[validation]\nsource = \"linked\"\nline-coverage = 100\nbranch-coverage = 100\n{extra_validation}"
        ),
    )
    .unwrap();
    symlink("../examples", package.join("linked")).unwrap();
    (root, examples, package)
}

#[cfg(unix)]
fn find_report(directory: &std::path::Path) -> Option<PathBuf> {
    for entry in fs::read_dir(directory).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(report) = find_report(&path) {
                return Some(report);
            }
        } else if path.file_name().and_then(|name| name.to_str())
            == Some("examples-validation.json")
        {
            return Some(path);
        }
    }
    None
}

#[cfg(unix)]
#[test]
fn validation_rejects_a_copied_example_directory() {
    let (root, _, package) =
        validation_fixture("copied-validation", "test:\n    assert(true)\n", "");
    fs::remove_file(package.join("linked")).unwrap();
    fs::create_dir(package.join("linked")).unwrap();
    fs::write(
        package.join("linked/example.sev"),
        "test:\n    assert(true)\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("must be a relative symlink"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_fails_when_a_line_is_uncovered() {
    let (root, _, package) = validation_fixture(
        "line-coverage",
        "def never_called():\n    value := 1\n\ntest:\n    assert(true)\n",
        "",
    );
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("line coverage"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_fails_when_a_branch_is_uncovered() {
    let (root, _, package) = validation_fixture(
        "branch-coverage",
        "def choose() -> int:\n    if true:\n        return 1\n    else:\n        return 2\n\ntest:\n    assert(choose() == 1)\n",
        "",
    );
    fs::write(
        package.join("validation.toml"),
        "[validation]\nsource = \"linked\"\nline-coverage = 0\nbranch-coverage = 100\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("branch coverage"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_fails_on_exact_stdout_mismatch() {
    let (root, examples, package) = validation_fixture(
        "stdout-mismatch",
        "def main():\n    print(\"actual\")\n",
        "",
    );
    fs::write(examples.join("example.stdout"), "expected\n").unwrap();
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("stdout did not match"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_reports_canonical_paths_and_structured_routes() {
    let (root, examples, package) = validation_fixture(
        "canonical-report",
        "test:\n    assert(true)\n",
        "\n[[example]]\npath = \"linked/example.sev\"\nrequired-routes = [\"standard\"]\nallow-fallback = false\n",
    );
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report = fs::read_to_string(find_report(&package).expect("validation report")).unwrap();
    let canonical = fs::canonicalize(examples.join("example.sev")).unwrap();
    assert!(
        report.contains(&canonical.display().to_string()),
        "{report}"
    );
    assert!(report.contains("\"routes\": [\"standard\"]"), "{report}");
    assert!(
        !report.contains("validation/linked/example.sev"),
        "{report}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn dependency_tests_are_not_executed_as_example_tests() {
    let (root, examples, package) = validation_fixture(
        "dependency-test-isolation",
        "import support\n\ntest \"example test\":\n    assert(true)\n",
        "",
    );
    let dependency = root.join("support");
    fs::create_dir_all(dependency.join("src")).unwrap();
    fs::write(
        dependency.join("package.toml"),
        "[package]\nname = \"support\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    fs::write(
        dependency.join("src/lib.sev"),
        "test \"dependency test\":\n    assert(false)\n",
    )
    .unwrap();
    fs::write(
        package.join("package.toml"),
        "[package]\nname = \"examples-validation\"\npublish = false\n\n[lib]\npath = \"src/validation.sev\"\n\n[dependencies]\nsupport = { path = \"../support\" }\n",
    )
    .unwrap();
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("test example test ... ok"), "{stdout}");
    assert!(!stdout.contains("dependency test"), "{stdout}");
    assert!(fs::canonicalize(examples).is_ok());
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn validation_can_expect_a_nonzero_example_exit() {
    let (root, _, package) = validation_fixture(
        "expected-exit",
        "def main():\n    assert(false)\n",
        "\n[[example]]\npath = \"linked/example.sev\"\nexpected-exit = 1\n",
    );
    let output = sev().args(["test"]).arg(&package).output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}
