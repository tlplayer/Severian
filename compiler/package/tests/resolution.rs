use severian_package::{
    load_path_dependency_interfaces, publish_package, resolve_dependencies, update_dependencies,
};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time follows the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-registry-resolution-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn published_versions_are_resolved_verified_cached_locked_and_importable() {
    let _environment = ENVIRONMENT.lock().unwrap();
    let root = temporary_directory();
    let registry = root.join("registry");
    let cache = root.join("consumer-home");
    let tensor = root.join("tensor-source");
    let application = root.join("application");
    std::fs::create_dir_all(tensor.join("src")).unwrap();
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::write(
        tensor.join("package.toml"),
        "[package]\nname = \"tensor-fixture\"\nversion = \"0.8.4\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        tensor.join("src/lib.sev"),
        "def answer() -> int:\n    return 42\n",
    )
    .unwrap();
    let published = publish_package(&tensor.join("package.toml"), Some(&registry)).unwrap();
    assert_eq!(published.version, "0.8.4");
    assert_eq!(published.checksum.as_deref().map(str::len), Some(64));

    std::fs::write(
        application.join("package.toml"),
        format!(
            "[package]\nname = \"application\"\nversion = \"0.1.0\"\n\n[dependencies]\ntensor = {{ package = \"tensor-fixture\", version = \"0.8\", registry = {:?} }}\n",
            registry.display().to_string()
        ),
    )
    .unwrap();
    std::fs::write(
        application.join("src/main.sev"),
        "from tensor import answer\n\ndef main():\n    print(answer())\n",
    )
    .unwrap();

    let previous_home = std::env::var_os("SEVERIAN_HOME");
    std::env::set_var("SEVERIAN_HOME", &cache);
    let resolution = resolve_dependencies(&application.join("package.toml")).unwrap();
    assert_eq!(resolution.dependencies.len(), 1);
    let dependency = &resolution.dependencies[0];
    assert_eq!(dependency.import_name, "tensor");
    assert_eq!(dependency.package_name, "tensor-fixture");
    assert_eq!(dependency.version, "0.8.4");
    assert_eq!(dependency.root, cache.join("packages/tensor-fixture/0.8.4"));
    let lock = std::fs::read_to_string(application.join("sev.lock")).unwrap();
    assert!(lock.contains("name = \"tensor-fixture\""));
    assert!(lock.contains("version = \"0.8.4\""));
    assert!(lock.contains("checksum = \""));

    let interfaces = load_path_dependency_interfaces(&application.join("package.toml")).unwrap();
    assert_eq!(interfaces.len(), 1);
    assert_eq!(interfaces[0].name, "tensor");

    // Normal resolution honors the lock; an explicit update selects the newest
    // compatible release and replaces the exact lock selection.
    std::fs::write(
        tensor.join("package.toml"),
        "[package]\nname = \"tensor-fixture\"\nversion = \"0.8.5\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    publish_package(&tensor.join("package.toml"), Some(&registry)).unwrap();
    assert_eq!(
        resolve_dependencies(&application.join("package.toml"))
            .unwrap()
            .dependencies[0]
            .version,
        "0.8.4"
    );
    assert_eq!(
        update_dependencies(&application.join("package.toml"))
            .unwrap()
            .dependencies[0]
            .version,
        "0.8.5"
    );

    // A modified cache is discarded and recreated from the verified registry
    // source before it is exposed to the compiler.
    std::fs::write(
        cache.join("packages/tensor-fixture/0.8.5/src/lib.sev"),
        "def answer() -> int:\n    return 0\n",
    )
    .unwrap();
    resolve_dependencies(&application.join("package.toml")).unwrap();
    assert!(
        std::fs::read_to_string(cache.join("packages/tensor-fixture/0.8.5/src/lib.sev"))
            .unwrap()
            .contains("return 42")
    );

    if let Some(home) = previous_home {
        std::env::set_var("SEVERIAN_HOME", home);
    } else {
        std::env::remove_var("SEVERIAN_HOME");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn registry_tampering_is_rejected_before_caching() {
    let _environment = ENVIRONMENT.lock().unwrap();
    let root = temporary_directory();
    let registry = root.join("registry");
    let package = root.join("source");
    let application = root.join("application");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::create_dir_all(&application).unwrap();
    std::fs::write(
        package.join("package.toml"),
        "[package]\nname = \"verified\"\nversion = \"1.0.0\"\n\n[lib]\npath = \"src/lib.sev\"\n",
    )
    .unwrap();
    std::fs::write(
        package.join("src/lib.sev"),
        "def value() -> int:\n    return 1\n",
    )
    .unwrap();
    publish_package(&package.join("package.toml"), Some(&registry)).unwrap();
    std::fs::write(
        registry.join("packages/verified/1.0.0/src/lib.sev"),
        "def value() -> int:\n    return 999\n",
    )
    .unwrap();
    std::fs::write(
        application.join("package.toml"),
        format!(
            "[package]\nname = \"consumer\"\nversion = \"0.1.0\"\n\n[dependencies]\nverified = {{ version = \"1\", registry = {:?} }}\n",
            registry.display().to_string()
        ),
    )
    .unwrap();
    let error = resolve_dependencies(&application.join("package.toml")).unwrap_err();
    assert!(error.to_string().contains("checksum mismatch"));
    std::fs::remove_dir_all(root).unwrap();
}
