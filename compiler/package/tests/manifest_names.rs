use severian_package::nearest_manifest;
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must follow the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-manifest-names-{}-{nonce}",
        std::process::id()
    ))
}

#[test]
fn package_toml_is_discovered_from_nested_directories() {
    let directory = temporary_directory();
    let nested = directory.join("src/nested");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(
        directory.join("package.toml"),
        "[package]\nname = \"current\"\n",
    )
    .unwrap();

    let manifest = nearest_manifest(&nested).unwrap();
    assert_eq!(manifest, directory.join("package.toml"));

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn old_manifest_names_are_not_discovered() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("sev.toml"), "[package]\nname = \"old\"\n").unwrap();
    std::fs::write(
        directory.join("Severian.toml"),
        "[package]\nname = \"older\"\n",
    )
    .unwrap();

    assert!(nearest_manifest(&directory).is_none());

    let _ = std::fs::remove_dir_all(directory);
}
