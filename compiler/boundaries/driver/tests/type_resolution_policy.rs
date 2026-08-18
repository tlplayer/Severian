use severian_driver::check_path;
use std::time::{SystemTime, UNIX_EPOCH};

fn strict_package(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "severian-type-resolution-{label}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        concat!(
            "[package]\nname = \"strict\"\nversion = \"0.1.0\"\n",
            "[compiler.type_resolution]\n",
            "deny_any = true\n",
            "deny_tensor_any = true\n",
            "deny_unresolved = true\n",
            "deny_inferred_fallback = true\n",
            "deny_lost_type_information = true\n",
        ),
    )
    .unwrap();
    let source = root.join("src/main.sev");
    (root, source)
}

#[test]
fn strict_policy_rejects_inference_fallback_before_mir() {
    let (root, source) = strict_package("fallback");
    std::fs::write(&source, "def identity(value) -> Any:\n    return value\n").unwrap();

    let error = check_path(&source).unwrap_err().to_string();
    assert!(
        error.contains("error[E000207]: unresolved type escaped semantic analysis"),
        "{error}"
    );
    assert!(error.contains("InferenceFallback"));
    assert!(error.contains("compiler.type_resolution.deny_inferred_fallback = true"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn strict_policy_preserves_explicit_source_any() {
    let (root, source) = strict_package("explicit");
    std::fs::write(
        &source,
        "def identity(value: Any) -> Any:\n    return value\n",
    )
    .unwrap();

    check_path(&source).unwrap();
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn strict_policy_rejects_lost_collection_element_types() {
    let (root, source) = strict_package("lost");
    std::fs::write(
        &source,
        "def first(values: list[int]) -> Any:\n    return values[0]\n",
    )
    .unwrap();

    let error = check_path(&source).unwrap_err().to_string();
    assert!(error.contains("LostTypeInformation"));
    assert!(error.contains("compiler.type_resolution.deny_lost_type_information = true"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn strict_policy_rejects_unresolved_declared_types() {
    let (root, source) = strict_package("unresolved");
    std::fs::write(
        &source,
        "def consume(value: MissingType):\n    print(value)\n",
    )
    .unwrap();

    let error = check_path(&source).unwrap_err().to_string();
    assert!(error.contains("UnresolvedType"));
    assert!(error.contains("compiler.type_resolution.deny_unresolved = true"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn tensor_any_guard_rejects_an_unbound_tensor_generic() {
    let (root, source) = strict_package("tensor-any");
    std::fs::write(
        root.join("package.toml"),
        concat!(
            "[package]\nname = \"strict\"\nversion = \"0.1.0\"\n",
            "[compiler.type_resolution]\n",
            "deny_tensor_any = true\n",
        ),
    )
    .unwrap();
    std::fs::write(
        &source,
        concat!(
            "def tensor_identity[type](value: Tensor[type]) -> Tensor[type]:\n",
            "    return value\n\n",
            "def erase(value: Any) -> Any:\n",
            "    return tensor_identity(value)\n",
        ),
    )
    .unwrap();

    let error = check_path(&source).unwrap_err().to_string();
    assert!(error.contains("Tensor[Any]"), "{error}");
    assert!(error.contains("UnresolvedGeneric"), "{error}");
    assert!(
        error.contains("compiler.type_resolution.deny_tensor_any = true"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(root);
}
