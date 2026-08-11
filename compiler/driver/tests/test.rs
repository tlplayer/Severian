use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

static NATIVE_CLI_LOCK: Mutex<()> = Mutex::new(());

fn native_cli_lock() -> std::sync::MutexGuard<'static, ()> {
    NATIVE_CLI_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

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
    let _lock = native_cli_lock();
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
    let _lock = native_cli_lock();
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
    let _lock = native_cli_lock();
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

#[test]
fn build_uses_the_manifest_name_and_target_debug_directory() {
    let root = std::env::temp_dir().join(format!("severian-build-test-{}", std::process::id()));
    let source_directory = root.join("src");
    std::fs::create_dir_all(&source_directory).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"native-demo\"\nversion = \"0.1.0\"\nedition = \"2026\"\n",
    )
    .unwrap();
    std::fs::write(
        source_directory.join("main.sev"),
        "def main():\n    print(42)\n",
    )
    .unwrap();

    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .current_dir(&root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let executable = root.join("target/debug/native-demo");
    let output = Command::new(&executable).output().unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "42\n");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn build_compiles_library_artifacts_before_consuming_them() {
    let root = std::env::temp_dir().join(format!(
        "severian-vertical-build-test-{}",
        std::process::id()
    ));
    let application = root.join("application");
    let helper = root.join("helper");
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::create_dir_all(helper.join("src")).unwrap();
    std::fs::write(
        helper.join("package.toml"),
        "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        helper.join("src/lib.sev"),
        "def double(value: int) -> int:\n    return value * 2\n\ntest \"library-local\":\n    assert(double(3) == 6)\n",
    )
    .unwrap();
    std::fs::write(
        application.join("package.toml"),
        "[package]\nname = \"vertical-app\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[[bin]]\nname = \"vertical-app\"\npath = \"src/main.sev\"\n\n[dependencies]\nhelper = { path = \"../helper\", version = \"0.1.0\" }\n",
    )
    .unwrap();
    std::fs::write(
        application.join("src/main.sev"),
        "import helper\nfrom helper import double\n\ndef main():\n    print(helper.double(21))\n\ntest \"application-only\":\n    assert(double(4) == 8)\n",
    )
    .unwrap();

    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .current_dir(&application)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let artifact = helper.join("target/debug/deps/libhelper.sevi");
    let artifact_source = std::fs::read_to_string(&artifact).unwrap();
    assert!(artifact_source.contains("severian-library-artifact v1"));
    assert!(!artifact_source.contains("library-local"));
    let output = Command::new(application.join("target/debug/vertical-app"))
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "42\n");

    let test_binary = application.join("target/debug/vertical-app-tests");
    let compiled_tests = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("compile-tests")
        .arg("src/main.sev")
        .arg("-o")
        .arg(&test_binary)
        .current_dir(&application)
        .output()
        .unwrap();
    assert!(compiled_tests.status.success());
    assert!(String::from_utf8_lossy(&compiled_tests.stdout).contains("(1 native tests)"));
    let tests = Command::new(test_binary).output().unwrap();
    assert!(tests.status.success());
    assert_eq!(String::from_utf8(tests.stdout).unwrap(), "1 passed\n");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn assurance_commands_produce_coverage_mutation_and_memory_results() {
    let _lock = native_cli_lock();
    let root = std::env::temp_dir().join(format!("severian-assurance-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("boundary.sev");
    std::fs::write(
        &source,
        "def boundary(value: int) -> bool:\n    return value > 10\n\ntest:\n    assert(boundary(11))\n    assert(not boundary(10))\n",
    )
    .unwrap();

    let coverage = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("coverage")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        coverage.status.success(),
        "{}",
        String::from_utf8_lossy(&coverage.stderr)
    );
    let coverage_stdout = String::from_utf8_lossy(&coverage.stdout);
    assert!(coverage_stdout.contains("Lines      100.00%"));
    assert!(root.join("target/coverage/coverage-report.json").is_file());

    let mutation = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["test", "--mutate", "--limit", "1"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(mutation.status.success());
    let mutation_stdout = String::from_utf8_lossy(&mutation.stdout);
    assert!(mutation_stdout.contains("Killed:            1"));
    assert!(mutation_stdout.contains("Mutation score:    100.0%"));

    let memory = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("memory")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        memory.status.success(),
        "{}",
        String::from_utf8_lossy(&memory.stderr)
    );
    assert!(String::from_utf8_lossy(&memory.stdout).contains("0 target(s) with findings"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_tests_lower_a_typed_sorted_reverse_flag_as_a_boolean() {
    let _lock = native_cli_lock();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/03-collections-iteration/05-expressive-collections.sev");
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("test")
        .arg(fixture)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("4 passed"));
}
