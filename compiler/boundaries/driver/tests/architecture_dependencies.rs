use severian_package::BuildPolicy;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-architecture-dependencies-{}-{nonce}",
        std::process::id()
    ))
}

fn write_crate(root: &Path, name: &str, dependency: Option<&str>) {
    let directory = root.join("crates").join(name);
    std::fs::create_dir_all(directory.join("src")).unwrap();
    let dependency = dependency.map_or(String::new(), |dependency| {
        format!("\n[dependencies]\n{dependency} = {{ path = \"../{dependency}\" }}\n")
    });
    std::fs::write(
        directory.join("Cargo.toml"),
        format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n{dependency}"),
    )
    .unwrap();
    std::fs::write(directory.join("src/lib.rs"), "").unwrap();
}

#[test]
fn dependency_pass_reports_cycles_layers_unknown_packages_and_explicit_rules() {
    let root = temporary_directory();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("package.toml"),
        r#"
[workspace]
members = []

[architecture]
deny_cycles = true
deny_unknown_layers = true
deny_layer_violations = true

[architecture.layers]
include = ["crates/*"]
order = ["core", "app"]

[[architecture.rule]]
from = "crates/app/**"
deny = ["crates/core/**"]

[architecture.files]
include = []
"#,
    )
    .unwrap();
    write_crate(&root, "core", Some("app"));
    write_crate(&root, "app", Some("core"));
    write_crate(&root, "extra", None);

    let policy = BuildPolicy::for_input(&root).unwrap();
    let analysis = severian_driver::architecture::analyze_dependencies(&policy).unwrap();
    let codes = analysis
        .findings
        .iter()
        .map(|finding| finding.code)
        .collect::<Vec<_>>();
    assert!(codes.contains(&"architecture::dependency_cycle"));
    assert!(codes.contains(&"architecture::backward_layer_dependency"));
    assert!(codes.contains(&"architecture::unknown_layer"));
    assert!(codes.contains(&"architecture::forbidden_dependency"));
    let cycle = analysis
        .findings
        .iter()
        .find(|finding| finding.code == "architecture::dependency_cycle")
        .unwrap();
    assert!(cycle.message.contains("app"));
    assert!(cycle.message.contains("core"));
    assert!(cycle.line.is_some());
    assert!(analysis.to_dot().contains("digraph severian_architecture"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn repository_compiler_dependency_graph_respects_declared_boundaries() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let policy = BuildPolicy::for_input(&root).unwrap();
    let analysis = severian_driver::architecture::analyze_dependencies(&policy).unwrap();
    assert!(
        analysis.findings.is_empty(),
        "{}",
        analysis
            .findings
            .iter()
            .map(|finding| format!("{}: {}", finding.code, finding.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn sev_check_runs_the_architecture_pass_before_succeeding() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        r#"
[package]
name = "architecture-check"
version = "0.1.0"

[[bin]]
name = "architecture-check"
path = "src/main.sev"

[architecture]
deny_cycles = true
deny_layer_violations = true

[architecture.layers]
include = ["crates/*"]
order = ["core", "app"]

[architecture.files]
include = []
"#,
    )
    .unwrap();
    std::fs::write(root.join("src/main.sev"), "def main():\n    print(1)\n").unwrap();
    write_crate(&root, "core", Some("app"));
    write_crate(&root, "app", None);

    let checked = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    assert!(!checked.status.success());
    let stdout = String::from_utf8_lossy(&checked.stdout);
    let stderr = String::from_utf8_lossy(&checked.stderr);
    assert!(
        stdout.contains("architecture::backward_layer_dependency"),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("core"));
    assert!(stdout.contains("app"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn architecture_graph_command_emits_standalone_dot() {
    let root = temporary_directory();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("package.toml"),
        r#"
[workspace]
members = []

[architecture]
deny_cycles = true
deny_layer_violations = true

[architecture.layers]
include = ["crates/*"]
order = ["core", "app"]

[architecture.files]
include = []
"#,
    )
    .unwrap();
    write_crate(&root, "core", None);
    write_crate(&root, "app", Some("core"));

    let graphed = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("architecture")
        .arg(&root)
        .arg("--graph")
        .output()
        .unwrap();
    assert!(
        graphed.status.success(),
        "{}",
        String::from_utf8_lossy(&graphed.stderr)
    );
    let stdout = String::from_utf8_lossy(&graphed.stdout);
    assert!(stdout.starts_with("digraph severian_architecture {"));
    assert!(!stdout.contains("Architecture\n"));
    assert!(stdout.contains("app"));
    assert!(stdout.contains("core"));
    let _ = std::fs::remove_dir_all(root);
}
