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
fn direct_source_invocation_resolves_project_relative_imports() {
    let _lock = native_cli_lock();
    let root =
        std::env::temp_dir().join(format!("severian-local-import-test-{}", std::process::id()));
    std::fs::create_dir_all(root.join("local")).unwrap();
    std::fs::write(
        root.join("helpers.sev"),
        "def double(value: int) -> int:\n    return value * 2\n",
    )
    .unwrap();
    std::fs::write(
        root.join("local/math.sev"),
        "def increment(value: int) -> int:\n    return value + 1\n",
    )
    .unwrap();
    let main = root.join("main.sev");
    std::fs::write(
        &main,
        concat!(
            "import \"helpers.sev\"\n",
            "import \"local/math\" as local_math\n",
            "\n",
            "def main():\n",
            "    print(helpers.double(local_math.increment(20)))\n",
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg(&main)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "42\n");
    std::fs::remove_dir_all(root).unwrap();
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
        "def main():\n    print(42)\n\ntest \"entrypoint\":\n    main()\n",
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
fn build_is_blocked_below_seventy_five_percent_line_coverage() {
    let root = std::env::temp_dir().join(format!(
        "severian-build-coverage-gate-test-{}",
        std::process::id()
    ));
    let source_directory = root.join("src");
    std::fs::create_dir_all(&source_directory).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"uncovered-demo\"\nversion = \"0.1.0\"\nedition = \"2026\"\n",
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

    assert!(!build.status.success());
    assert!(String::from_utf8_lossy(&build.stderr).contains("required threshold is 75.00%"));
    assert!(!root.join("target/debug/uncovered-demo").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn init_exposes_the_complete_lenient_manifest_contract() {
    let root = std::env::temp_dir().join(format!("severian-init-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let initialized = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("init")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let manifest = std::fs::read_to_string(root.join("package.toml")).unwrap();
    assert!(manifest.contains("diagnostics = \"user\""));
    assert!(manifest.contains("pipeline = ["));
    assert!(manifest.contains("[profile.dev]"));
    assert!(manifest.contains("[architecture]"));
    assert!(manifest.contains("deny_cycles = true"));
    assert!(manifest.contains("[architecture.files]"));
    assert!(manifest.contains("# [workspace]"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn new_creates_and_runs_a_native_severian_project() {
    let _lock = native_cli_lock();
    let root = std::env::temp_dir().join(format!("severian-new-test-{}", std::process::id()));
    let created = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("new")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stderr)
    );
    assert!(root.join("package.toml").is_file());
    assert!(root.join("sev.lock").is_file());
    let manifest = std::fs::read_to_string(root.join("package.toml")).unwrap();
    assert!(manifest.contains("diagnostics = \"user\""));
    assert!(manifest.contains("[compiler.type_resolution]"));
    assert!(manifest.contains("deny_inferred_fallback = false"));
    assert!(manifest.contains("minimum = 0"));
    assert!(manifest.contains("leaks = \"allow\""));
    assert!(manifest.contains("[profile.release]"));
    assert!(manifest.contains("# license = \"MIT\""));

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&root)
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

    let build = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("build")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
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
        "import helper\nfrom helper import double\n\ndef main():\n    print(helper.double(21))\n\ntest \"application-only\":\n    assert(double(4) == 8)\n    main()\n",
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
    assert_eq!(String::from_utf8(tests.stdout).unwrap(), "42\n1 passed\n");
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
    assert!(root.join("target/coverage/coverage.hits").is_file());

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
fn repository_coverage_ignores_generated_work_and_expected_negative_fixtures() {
    let _lock = native_cli_lock();
    let root = std::env::temp_dir().join(format!(
        "severian-coverage-discovery-test-{}",
        std::process::id()
    ));
    let negative = root.join("docs/examples/bugs/ownership/sample");
    let generated = root.join("bench/.work/generated");
    std::fs::create_dir_all(&negative).unwrap();
    std::fs::create_dir_all(&generated).unwrap();
    std::fs::write(
        root.join("covered.sev"),
        "def covered() -> int:\n    return 1\n\ntest:\n    assert(covered() == 1)\n",
    )
    .unwrap();
    std::fs::write(negative.join("invalid.sev"), "this cannot parse").unwrap();
    std::fs::write(generated.join("broken.sev"), "this cannot parse").unwrap();

    let coverage = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("coverage")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        coverage.status.success(),
        "{}",
        String::from_utf8_lossy(&coverage.stderr)
    );
    assert!(String::from_utf8_lossy(&coverage.stdout).contains("1 test(s) across 1 target(s)"));
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

#[test]
fn triple_quoted_block_strings_compile_and_preserve_newlines() {
    let _lock = native_cli_lock();
    let root =
        std::env::temp_dir().join(format!("severian-block-string-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        "def main():\n    message = \"\"\"first line\nsecond line\n\"\"\"\n    print(message)\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "first line\nsecond line\n\n"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn formatted_triple_quoted_block_strings_compile_and_interpolate() {
    let _lock = native_cli_lock();
    let root = std::env::temp_dir().join(format!(
        "severian-formatted-block-string-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("main.sev");
    std::fs::write(
        &source,
        concat!(
            "def describe(name: string, version: int) -> string:\n",
            "    return f\"\"\"model {name}\nversion {version}\n\"\"\"\n",
            "\n",
            "def main():\n",
            "    print(describe(\"Qwen\", 3))\n",
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "model Qwen\nversion 3\n\n"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn strict_type_resolution_rejects_inferred_any_with_actionable_source_context() {
    let root = std::env::temp_dir().join(format!(
        "severian-type-resolution-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"strict-example\"\nversion = \"0.1.0\"\n\n[compiler.type_resolution]\ndeny_any = true\ndeny_inferred_fallback = true\n\n[[bin]]\nname = \"strict-example\"\npath = \"src/main.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.sev"),
        "def identity(value) -> Any:\n    return value\n",
    )
    .unwrap();

    let rejected = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let error = String::from_utf8_lossy(&rejected.stderr);
    assert!(error.contains("E000207"));
    assert!(error.contains("parameter `value` has no declared type"));
    assert!(error.contains("deny_inferred_fallback = true"));

    std::fs::write(
        root.join("src/main.sev"),
        "def identity(value: Any) -> Any:\n    return value\n",
    )
    .unwrap();
    let explicit = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );

    std::fs::write(
        root.join("src/main.sev"),
        "import \"helpers.sev\" as helpers\n\ndef main():\n    print(helpers.identity(1))\n",
    )
    .unwrap();
    std::fs::write(
        root.join("helpers.sev"),
        "def identity(value) -> Any:\n    return value\n",
    )
    .unwrap();
    let local_module = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    assert!(!local_module.status.success());
    let error = String::from_utf8_lossy(&local_module.stderr);
    assert!(error.contains("helpers.sev"));
    assert!(error.contains("E000207"));
    assert!(error.contains("parameter `value` has no declared type"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runs_nested_functions_and_python_style_match() {
    let _lock = native_cli_lock();
    let source_path = std::env::temp_dir().join(format!(
        "severian-nested-match-test-{}.sev",
        std::process::id()
    ));
    std::fs::write(
        &source_path,
        concat!(
            "def outer(value: int) -> int:\n",
            "    def inner(offset: int) -> int:\n",
            "        return value + offset\n",
            "    return inner(2)\n",
            "\n",
            "def nested_message() -> string:\n",
            "    def inner() -> string:\n",
            "        return \"nested\"\n",
            "    return inner()\n",
            "\n",
            "def classify(extension: string) -> string:\n",
            "    match extension:\n",
            "        case \".yaml\" | \".yml\":\n",
            "            return \"yaml\"\n",
            "        case _:\n",
            "            return \"other\"\n",
            "\n",
            "def main():\n",
            "    print(outer(40))\n",
            "    print(nested_message())\n",
            "    print(classify(\".yml\"))\n",
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("run")
        .arg(&source_path)
        .output()
        .unwrap();
    std::fs::remove_file(source_path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "42\nnested\nyaml\n"
    );
}
