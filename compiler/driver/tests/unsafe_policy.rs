use severian_driver::compile_path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time follows the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-unsafe-policy-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn native_abi_cannot_be_exempted_for_an_application_binary() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"unsafe-app\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"native-abi\"]\nsources = [\"src/main.sev\"]\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.sev"),
        "unsafe:\n    native(\"host_value\") def hostValue() -> int\n\ndef main():\n    print(hostValue())\n",
    )
    .unwrap();

    let error = compile_path(&root.join("src/main.sev")).unwrap_err();
    assert!(error.to_string().contains("unsafe policy"));
    assert!(error.to_string().contains("native-abi"));
    assert!(error.to_string().contains("non-library target"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn libraries_must_explicitly_opt_in_to_unsafe_code() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("src")).unwrap();
    let manifest = root.join("package.toml");
    let source = root.join("src/lib.sev");
    std::fs::write(
        &manifest,
        "[package]\nname = \"native-library\"\nversion = \"0.1.0\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        &source,
        "unsafe:\n    native(\"host_value\") def hostValue() -> int\n",
    )
    .unwrap();

    let error = compile_path(&source).unwrap_err();
    assert!(error.to_string().contains("native-abi"));
    std::fs::write(
        &manifest,
        "[package]\nname = \"native-library\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"native-abi\"]\nsources = [\"src/lib.sev\"]\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    compile_path(&source).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn opted_in_libraries_still_reject_unsafe_test_bodies() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"tested-library\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"unsafe-blocks\"]\nsources = [\"src/lib.sev\"]\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    let source = root.join("src/lib.sev");
    std::fs::write(
        &source,
        "def probe():\n    return\n\ntest \"safe only\":\n    unsafe:\n        value = 1\n",
    )
    .unwrap();

    let error = compile_path(&source).unwrap_err();
    assert!(error
        .to_string()
        .contains("tests may not contain `unsafe` blocks"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn pointer_examples_can_receive_a_source_scoped_capability() {
    let root = temporary_directory();
    std::fs::create_dir_all(root.join("examples")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"pointer-example\"\nversion = \"0.1.0\"\n\n[package.unsafe]\ncapabilities = [\"pointers\"]\nsources = [\"examples/pointer.sev\"]\n",
    )
    .unwrap();
    let allowed = root.join("examples/pointer.sev");
    let denied = root.join("examples/not-listed.sev");
    let source = "def first(values: list[int]) -> int:\n    unsafe:\n        pointer = &values\n        return pointer[0]\n";
    std::fs::write(&allowed, source).unwrap();
    std::fs::write(&denied, source).unwrap();

    compile_path(&allowed).unwrap();
    let error = compile_path(&denied).unwrap_err();
    assert!(error.to_string().contains("source-file"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn qwen_packages_use_safe_tensor_and_platform_apis() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let qwen_sources = [
        "benchmarks/inference/severian/qwen_kernels.sev",
        "benchmarks/inference/severian/qwen_prefill_kernels.sev",
        "benchmarks/inference/severian/qwen_decode_kernels.sev",
        "benchmarks/inference/severian/qwen_tokenizer.sev",
    ];
    let qwen_manifests = [
        "benchmarks/inference/severian/modules/qwen_kernels/package.toml",
        "benchmarks/inference/severian/modules/qwen_prefill_kernels/package.toml",
        "benchmarks/inference/severian/modules/qwen_decode_kernels/package.toml",
        "benchmarks/inference/severian/modules/qwen_tokenizer/package.toml",
    ];

    for relative in qwen_sources {
        let source = std::fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(!source.contains("unsafe:"), "{relative} must remain safe");
        assert!(
            !source.contains("native("),
            "{relative} must use library APIs"
        );
    }
    for relative in qwen_manifests {
        let manifest = std::fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(
            !manifest.contains("[package.unsafe]"),
            "{relative} must not receive an unsafe capability"
        );
    }
}
