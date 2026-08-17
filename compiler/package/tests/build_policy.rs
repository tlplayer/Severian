use severian_package::{workspace_binary_targets, BuildPolicy, TypeResolutionPolicy};
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
pipeline = ["compile", "architecture", "test", "profile", "coverage", "memory", "integ"]

[coverage]
minimum = 99
regions = 99
branches = 99
functions = 99
per_file = true
exclude = ["src/os/**", "src/generated.sev"]

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
    assert_eq!(policy.coverage.minimum, 99.0);
    assert_eq!(policy.coverage.regions, Some(99.0));
    assert_eq!(policy.coverage.branches, Some(99.0));
    assert_eq!(policy.coverage.functions, Some(99.0));
    assert!(policy.coverage.per_file);
    assert_eq!(policy.coverage.exclude, ["src/os/**", "src/generated.sev"]);
    assert_eq!(policy.files.soft_lines, 400);
    assert_eq!(policy.files.exceptions.len(), 1);
    assert_eq!(
        policy.files.exceptions[0].owner.as_deref(),
        Some("compiler")
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn coverage_policy_rejects_unknown_and_mistyped_settings() {
    let root = temporary_directory("coverage-policy-invalid");
    std::fs::create_dir_all(&root).unwrap();
    let manifest = root.join("package.toml");
    std::fs::write(&manifest, "[coverage]\nfunctons = 99\n").unwrap();
    let error = BuildPolicy::for_input(&root).unwrap_err().to_string();
    assert!(error.contains("unknown `coverage` setting `functons`"));

    std::fs::write(&manifest, "[coverage]\nper_file = \"yes\"\n").unwrap();
    let error = BuildPolicy::for_input(&root).unwrap_err().to_string();
    assert!(error.contains("`coverage.per_file` must be a boolean"));

    std::fs::write(&manifest, "[coverage]\nexclude = [\"src/os/**\", 1]\n").unwrap();
    let error = BuildPolicy::for_input(&root).unwrap_err().to_string();
    assert!(error.contains("every `coverage.exclude` entry must be a string"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn architecture_graph_policy_is_typed_and_scoped() {
    let root = temporary_directory("architecture-graph-policy");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("package.toml"),
        r#"
[workspace]
members = []

[architecture]
enforce = true
deny_cycles = true
deny_unknown_layers = true
deny_layer_violations = true

[architecture.layers]
include = ["compiler/*"]
order = ["hir", "mir", "backend"]

[[architecture.rule]]
from = "compiler/mir/**"
allow = ["compiler/hir/**"]
deny = ["compiler/backend/**"]
"#,
    )
    .unwrap();

    let policy = BuildPolicy::for_input(&root).unwrap();
    assert_eq!(policy.architecture.layers.order, ["hir", "mir", "backend"]);
    assert_eq!(policy.architecture.layers.include, ["compiler/*"]);
    assert_eq!(policy.architecture.rules.len(), 1);
    assert_eq!(policy.architecture.rules[0].allow, ["compiler/hir/**"]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn architecture_policy_rejects_unknown_settings() {
    let root = temporary_directory("architecture-policy-typo");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[architecture]\ndeny_cylces = true\n",
    )
    .unwrap();

    let error = BuildPolicy::for_input(&root).unwrap_err().to_string();
    assert!(error.contains("unknown `architecture` setting `deny_cylces`"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn type_resolution_policy_loads_every_independent_guard() {
    let root = temporary_directory("type-resolution-policy");
    std::fs::create_dir_all(&root).unwrap();
    let manifest = root.join("package.toml");
    std::fs::write(
        &manifest,
        concat!(
            "[package]\nname = \"strict\"\n",
            "[compiler.type_resolution]\n",
            "deny_any = true\n",
            "deny_tensor_any = true\n",
            "deny_unresolved = true\n",
            "deny_inferred_fallback = true\n",
            "deny_lost_type_information = true\n",
        ),
    )
    .unwrap();

    assert_eq!(
        TypeResolutionPolicy::for_manifest(Some(&manifest)).unwrap(),
        TypeResolutionPolicy {
            deny_any: true,
            deny_tensor_any: true,
            deny_unresolved: true,
            deny_inferred_fallback: true,
            deny_lost_type_information: true,
        }
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn type_resolution_policy_rejects_misspelled_guards() {
    let root = temporary_directory("type-resolution-policy-typo");
    std::fs::create_dir_all(&root).unwrap();
    let manifest = root.join("package.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"strict\"\n[compiler.type_resolution]\ndeny_an = true\n",
    )
    .unwrap();

    let error = TypeResolutionPolicy::for_manifest(Some(&manifest)).unwrap_err();
    assert!(error
        .to_string()
        .contains("unknown `compiler.type_resolution` setting `deny_an`"));
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
