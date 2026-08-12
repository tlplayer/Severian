use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time follows the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-package-cli-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn published_dependency_builds_and_runs_without_a_source_path() {
    let root = temporary_directory();
    let registry = root.join("registry");
    let severian_home = root.join("sev-home");
    let library = root.join("answer-package");
    let application = root.join("application");
    std::fs::create_dir_all(library.join("src")).unwrap();
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::write(
        library.join("package.toml"),
        "[package]\nname = \"answer-package\"\nversion = \"1.4.2\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        library.join("src/lib.sev"),
        "def answer() -> int:\n    return 42\n",
    )
    .unwrap();
    severian_package::publish_package(&library.join("package.toml"), Some(&registry)).unwrap();

    std::fs::write(
        application.join("package.toml"),
        format!(
            "[package]\nname = \"registry-app\"\nversion = \"0.1.0\"\n\n[dependencies]\nanswer = {{ package = \"answer-package\", version = \"1.4\", registry = {:?} }}\n",
            registry.display().to_string()
        ),
    )
    .unwrap();
    std::fs::write(
        application.join("src/main.sev"),
        "from answer import answer\n\ndef main():\n    print(answer())\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["run", "."])
        .current_dir(&application)
        .env("SEVERIAN_HOME", &severian_home)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "42\n");
    let lock = std::fs::read_to_string(application.join("sev.lock")).unwrap();
    assert!(lock.contains("version = \"1.4.2\""));
    assert!(lock.contains("checksum = \""));
    assert!(severian_home
        .join("packages/answer-package/1.4.2/src/lib.sev")
        .is_file());
    assert!(!std::fs::read_to_string(application.join("src/main.sev"))
        .unwrap()
        .contains(&registry.display().to_string()));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn add_and_remove_manage_path_dependencies_and_the_lockfile() {
    let root = temporary_directory();
    let library = root.join("helper");
    let application = root.join("application");
    std::fs::create_dir_all(library.join("src")).unwrap();
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::write(
        library.join("package.toml"),
        "[package]\nname = \"helper\"\nversion = \"0.2.1\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        library.join("src/lib.sev"),
        "def value() -> int:\n    return 1\n",
    )
    .unwrap();
    std::fs::write(
        application.join("package.toml"),
        "[package]\nname = \"application\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    let add = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["add", "helper", "--path", "../helper", "--version", "0.2"])
        .current_dir(&application)
        .output()
        .unwrap();
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );
    assert!(std::fs::read_to_string(application.join("package.toml"))
        .unwrap()
        .contains("helper"));
    assert!(std::fs::read_to_string(application.join("sev.lock"))
        .unwrap()
        .contains("version = \"0.2.1\""));

    let remove = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["remove", "helper"])
        .current_dir(&application)
        .output()
        .unwrap();
    assert!(remove.status.success());
    assert!(!std::fs::read_to_string(application.join("package.toml"))
        .unwrap()
        .contains("helper"));
    assert!(!std::fs::read_to_string(application.join("sev.lock"))
        .unwrap()
        .contains("[[packages]]"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn install_builds_a_published_binary_into_severian_home() {
    let root = temporary_directory();
    let registry = root.join("registry");
    let severian_home = root.join("sev-home");
    let tool = root.join("tool");
    std::fs::create_dir_all(tool.join("src")).unwrap();
    std::fs::write(
        tool.join("package.toml"),
        "[package]\nname = \"answer-tool\"\nversion = \"2.1.3\"\n",
    )
    .unwrap();
    std::fs::write(tool.join("src/main.sev"), "def main():\n    print(42)\n").unwrap();
    severian_package::publish_package(&tool.join("package.toml"), Some(&registry)).unwrap();

    let install = Command::new(env!("CARGO_BIN_EXE_sev"))
        .args(["install", "answer-tool", "--version", "2.1"])
        .env("SEVERIAN_REGISTRY", &registry)
        .env("SEVERIAN_HOME", &severian_home)
        .output()
        .unwrap();
    assert!(
        install.status.success(),
        "{}",
        String::from_utf8_lossy(&install.stderr)
    );
    let output = Command::new(severian_home.join("bin/answer-tool"))
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "42\n");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn publish_writes_an_immutable_verified_registry_entry() {
    let root = temporary_directory();
    let registry = root.join("registry");
    let package = root.join("library");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(
        package.join("package.toml"),
        "[package]\nname = \"published-library\"\nversion = \"3.2.1\"\n",
    )
    .unwrap();
    std::fs::write(
        package.join("src/lib.sev"),
        "def value() -> int:\n    return 42\n",
    )
    .unwrap();

    let publish = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("publish")
        .current_dir(&package)
        .env("SEVERIAN_REGISTRY", &registry)
        .output()
        .unwrap();
    assert!(
        publish.status.success(),
        "{}",
        String::from_utf8_lossy(&publish.stderr)
    );
    assert!(registry
        .join("packages/published-library/3.2.1/src/lib.sev")
        .is_file());
    let checksum =
        std::fs::read_to_string(registry.join("checksums/published-library/3.2.1.sha256")).unwrap();
    assert_eq!(checksum.trim().len(), 64);

    let duplicate = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("publish")
        .current_dir(&package)
        .env("SEVERIAN_REGISTRY", &registry)
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("already published"));
    std::fs::remove_dir_all(root).unwrap();
}
