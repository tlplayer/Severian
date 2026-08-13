use severian_package::{workspace_binary_targets, BuildPolicy};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary_directory(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("severian-{label}-{}-{nonce}", std::process::id()))
}

#[test]
fn complete_pipeline_and_architecture_limits_load_from_manifest() {
    let root = temporary_directory("policy-manifest");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("package.toml"),
        r#"
[package]
name = "policy"

[build]
pipeline = ["compile", "architecture", "test", "profile", "coverage", "memory", "integration"]

[architecture.files]
soft_lines = 400
hard_lines = 700
include = ["src/**/*.sev"]

[[architecture.files.exception]]
path = "src/generated.sev"
hard_lines = 900
reason = "generated protocol table awaiting a data-file loader"
expires = "2099-01-01"
owner = "compiler"
"#,
    )
    .unwrap();

    let policy = BuildPolicy::for_input(&root).unwrap();
    assert_eq!(policy.pipeline.len(), 7);
    assert_eq!(policy.files.soft_lines, 400);
    assert_eq!(policy.files.exceptions.len(), 1);
    assert_eq!(
        policy.files.exceptions[0].owner.as_deref(),
        Some("compiler")
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn workspace_build_discovers_every_binary_and_library_in_nested_workspaces() {
    let root = temporary_directory("complete-workspace");
    let app = root.join("group/app");
    let library = root.join("group/library");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::create_dir_all(library.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[workspace]\nmembers = [\"group\"]\n",
    )
    .unwrap();
    std::fs::write(
        root.join("group/package.toml"),
        "[workspace]\nmembers = [\"app\", \"library\"]\n",
    )
    .unwrap();
    std::fs::write(
        app.join("package.toml"),
        "[package]\nname = \"app\"\n\n[[bin]]\nname = \"one\"\npath = \"src/one.sev\"\n\n[[bin]]\nname = \"two\"\npath = \"src/two.sev\"\n",
    )
    .unwrap();
    std::fs::write(app.join("src/one.sev"), "def main():\n    pass\n").unwrap();
    std::fs::write(app.join("src/two.sev"), "def main():\n    pass\n").unwrap();
    std::fs::write(
        library.join("package.toml"),
        "[package]\nname = \"library\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        library.join("src/lib.sev"),
        "def answer() -> int:\n    return 42\n",
    )
    .unwrap();

    let targets = workspace_binary_targets(&root).unwrap();
    let mut names = targets
        .iter()
        .map(|target| target.name.as_str())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["library-lib", "one", "two"]);
    assert!(targets.iter().all(|target| target.package_root == root));
    let _ = std::fs::remove_dir_all(root);
}
