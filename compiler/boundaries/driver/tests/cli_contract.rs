use std::fs;
use std::path::PathBuf;
use std::process::Command;
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
