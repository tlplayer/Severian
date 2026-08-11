use severian_package::{is_legacy_manifest, nearest_manifest};
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
fn package_toml_is_preferred_over_the_legacy_filename() {
    let directory = temporary_directory();
    let nested = directory.join("src/nested");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(
        directory.join("Severian.toml"),
        "[package]\nname = \"legacy\"\n",
    )
    .unwrap();
    std::fs::write(
        directory.join("package.toml"),
        "[package]\nname = \"current\"\n",
    )
    .unwrap();

    let manifest = nearest_manifest(&nested).unwrap();
    assert_eq!(manifest, directory.join("package.toml"));
    assert!(!is_legacy_manifest(&manifest));

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn legacy_manifest_remains_discoverable_during_migration() {
    let directory = temporary_directory();
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("Severian.toml"),
        "[package]\nname = \"legacy\"\n",
    )
    .unwrap();

    let manifest = nearest_manifest(&directory).unwrap();
    assert!(is_legacy_manifest(&manifest));

    let _ = std::fs::remove_dir_all(directory);
}
