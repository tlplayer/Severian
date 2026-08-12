use std::{
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_project(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-cli-surface-{name}-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn debug_builds_with_symbols_and_launches_the_configured_debugger() {
    let root = temporary_project("debug");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"debug-demo\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.sev"),
        "def main():\n    print(42)\n\ntest \"main\":\n    main()\n",
    )
    .unwrap();
    let debugger = root.join("debugger");
    std::fs::write(
        &debugger,
        "#!/bin/sh\ntest -x \"$1\" || exit 41\nprintf '%s\\n' \"$1\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&debugger).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&debugger, permissions).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("debug")
        .current_dir(&root)
        .env("SEVERIAN_DEBUGGER", &debugger)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let binary = root.join("target/debug/debug-demo");
    assert!(binary.is_file());
    assert!(String::from_utf8_lossy(&output.stdout).contains("target/debug/debug-demo"));

    let explicit = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("debug")
        .arg(root.join("src/main.sev"))
        .env("SEVERIAN_DEBUGGER", &debugger)
        .output()
        .unwrap();
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert!(root.join("target/debug/main").is_file());

    if let Ok(readelf) = Command::new("readelf")
        .args(["--sections"])
        .arg(&binary)
        .output()
    {
        if readelf.status.success() {
            assert!(
                String::from_utf8_lossy(&readelf.stdout).contains(".debug_info"),
                "debug build did not contain a .debug_info section"
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn clean_removes_only_the_project_target_directory() {
    let root = temporary_project("clean");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("target/debug")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"clean-demo\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(root.join("src/main.sev"), "def main():\n    return\n").unwrap();
    std::fs::write(root.join("keep.txt"), "keep").unwrap();
    std::fs::write(root.join("target/debug/generated"), "generated").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("clean")
        .current_dir(&root)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(!root.join("target").exists());
    assert_eq!(
        std::fs::read_to_string(root.join("keep.txt")).unwrap(),
        "keep"
    );
    std::fs::remove_dir_all(root).unwrap();
}
