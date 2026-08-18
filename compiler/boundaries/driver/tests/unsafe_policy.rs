use severian_driver::compile_path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn source_files_below(root: &std::path::Path, output: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            source_files_below(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

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
        "unsafe:\n    extern(\"host_value\") def hostValue() -> int\n\ndef main():\n    print(hostValue())\n",
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
        "unsafe:\n    extern(\"host_value\") def hostValue() -> int\n",
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
    let qwen_sources = ["library/model/architectures/qwen/src/lib.sev"];

    for relative in qwen_sources {
        let source = std::fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(!source.contains("unsafe:"), "{relative} must remain safe");
        assert!(
            !source.contains("extern("),
            "{relative} must use library APIs"
        );
    }
}

#[test]
fn omnivoice_has_no_architecture_specific_native_abi() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = Vec::new();
    for relative in [
        "compiler/backend/src",
        "compiler/lowering/src",
        "compiler/platform/src",
        "compiler/runtime/src",
    ] {
        source_files_below(&workspace.join(relative), &mut sources);
    }
    let forbidden = [
        "__sev_omnivoice_audio_embedding_single",
        "__sev_omnivoice_qwen3_layer_single",
        "__sev_omnivoice_audio_logits_single",
    ];
    for path in sources {
        let source = std::fs::read_to_string(&path).unwrap();
        for symbol in forbidden {
            assert!(
                !source.contains(symbol),
                "{} must not provide architecture-specific native symbol {symbol}",
                path.display(),
            );
        }
    }
}

#[test]
fn generic_compiler_layers_do_not_name_model_architectures() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = Vec::new();
    for relative in [
        "compiler/backend/src",
        "compiler/lowering/src",
        "compiler/platform/src",
        "compiler/runtime/src",
    ] {
        source_files_below(&workspace.join(relative), &mut sources);
    }
    for path in sources {
        let source = std::fs::read_to_string(&path).unwrap().to_ascii_lowercase();
        for architecture in [
            "omnivoice",
            "qwen",
            "higgs",
            "llama",
            "whisper",
            "mistral",
            "ollama",
        ] {
            assert!(
                !source.contains(architecture),
                "{} must not name model architecture `{architecture}`",
                path.display(),
            );
        }
    }
}

#[test]
fn lowering_cannot_implement_the_network_service() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = Vec::new();
    source_files_below(&workspace.join("compiler/lowering/src"), &mut sources);
    for path in sources {
        let source = std::fs::read_to_string(&path).unwrap();
        for forbidden in [
            "#include <sys/socket.h>",
            "struct sockaddr",
            "getaddrinfo(",
            "socket(AF_",
            "__sev_network_",
            "__sev_udp_",
            "network_source(",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} must emit generic external calls, not implement networking via `{forbidden}`",
                path.display(),
            );
        }
    }
    assert!(workspace
        .join("library/network/native/posix/network.c")
        .is_file());
    assert!(!workspace.join("compiler/platform/src/network.rs").exists());
}

#[test]
fn migrated_library_services_do_not_return_to_the_compiler_bridge() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut sources = Vec::new();
    for relative in [
        "compiler/backend/src",
        "compiler/lowering/src",
        "compiler/platform/src",
    ] {
        source_files_below(&workspace.join(relative), &mut sources);
    }
    for path in sources {
        let source = std::fs::read_to_string(&path).unwrap();
        for forbidden in [
            "__sev_math_",
            "__sev_random_",
            "__sev_environment_",
            "__sev_process_",
            "void *__sev_file_read(",
            "sev_abi_v1_file_",
            "__sev_regex_",
            "sev_abi_v1_regex_",
            "__sev_file_format_",
            "__sev_json_",
            "__sev_csv_",
            "#include <regex.h>",
            "regcomp(",
            "library/file",
            "library/process",
            "library/network",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} must leave `{forbidden}` to its owning library package",
                path.display(),
            );
        }
    }
    for provider in [
        "library/math/native/math.c",
        "library/random/native/random.c",
        "library/environment/native/posix/environment.c",
        "library/process/native/posix/process.c",
        "library/file/native/posix/file.c",
        "library/regex/native/posix/regex.c",
    ] {
        assert!(workspace.join(provider).is_file(), "missing {provider}");
    }
}
