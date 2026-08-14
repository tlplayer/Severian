use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn catalog_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for category in std::fs::read_dir(repository().join("docs/error")).unwrap() {
        let category = category.unwrap().path();
        if !category.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(category).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|extension| extension == "sev") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn expected_code(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.split_once('-'))
        .map(|(code, _)| code)
        .unwrap()
}

fn temporary_directory(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "severian-error-catalog-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn run_fixture(path: &Path, code: &str) -> Output {
    match code {
        "E000103" => run_invalid_dependency(path),
        "E000207" => run_type_resolution_package(path),
        code if code.starts_with("E0009") => Command::new(env!("CARGO_BIN_EXE_sev"))
            .arg(path)
            .output()
            .unwrap(),
        code if code.starts_with('W') => Command::new(env!("CARGO_BIN_EXE_sev"))
            .arg("lint")
            .arg(path)
            .output()
            .unwrap(),
        _ => Command::new(env!("CARGO_BIN_EXE_sev"))
            .arg("check")
            .arg(path)
            .output()
            .unwrap(),
    }
}

fn run_type_resolution_package(fixture: &Path) -> Output {
    let root = temporary_directory("type-resolution");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("package.toml"),
        "[package]\nname = \"catalog-type-resolution\"\nversion = \"0.1.0\"\n\n[compiler.type_resolution]\ndeny_any = true\ndeny_inferred_fallback = true\n\n[[bin]]\nname = \"catalog-type-resolution\"\npath = \"src/E000207-unresolved-type-escape.sev\"\n",
    )
    .unwrap();
    std::fs::copy(fixture, root.join("src/E000207-unresolved-type-escape.sev")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&root)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(root);
    output
}

fn run_invalid_dependency(fixture: &Path) -> Output {
    let root = temporary_directory("invalid-dependency");
    let application = root.join("application");
    let dependency = root.join("broken");
    std::fs::create_dir_all(application.join("src")).unwrap();
    std::fs::create_dir_all(dependency.join("src")).unwrap();
    std::fs::write(
        application.join("package.toml"),
        "[package]\nname = \"catalog-application\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"catalog-application\"\npath = \"src/main.sev\"\n\n[dependencies]\nbroken = { path = \"../broken\" }\n",
    )
    .unwrap();
    std::fs::write(
        application.join("src/main.sev"),
        "import broken\n\ndef main():\n    print(\"catalog\")\n",
    )
    .unwrap();
    std::fs::write(
        dependency.join("package.toml"),
        "[package]\nname = \"broken\"\nversion = \"0.1.0\"\n\n[lib]\npath = \"src/E000103-invalid-package-source.sev\"\n",
    )
    .unwrap();
    std::fs::copy(
        fixture,
        dependency.join("src/E000103-invalid-package-source.sev"),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_sev"))
        .arg("check")
        .arg(&application)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(root);
    output
}

fn first_catalog_code(output: &str) -> Option<&str> {
    output
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '-'))
        .find(|part| {
            matches!(
                (part.as_bytes().first(), part.len()),
                (Some(b'E'), 7) | (Some(b'W'), 4)
            ) && part.as_bytes()[1..].iter().all(u8::is_ascii_digit)
        })
}

#[test]
fn every_catalog_fixture_emits_its_named_diagnostic_with_context() {
    let files = catalog_files();
    assert!(!files.is_empty());
    for path in files {
        let code = expected_code(&path);
        let output = run_fixture(&path, code);
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if code.starts_with('E') {
            assert!(
                !output.status.success(),
                "{code} unexpectedly passed:\n{combined}"
            );
        } else {
            assert!(output.status.success(), "{code} lint failed:\n{combined}");
        }
        assert_eq!(
            first_catalog_code(&combined),
            Some(code),
            "{} emitted the wrong first diagnostic:\n{combined}",
            path.display()
        );
        assert!(
            combined.contains(path.file_name().unwrap().to_str().unwrap()),
            "{code} omitted its source filename:\n{combined}"
        );
        assert!(
            combined.contains(" --> "),
            "{code} omitted a location:\n{combined}"
        );
        assert!(
            combined.contains(" | "),
            "{code} omitted a snippet:\n{combined}"
        );

        let explanation = Command::new(env!("CARGO_BIN_EXE_sev"))
            .args(["explain", code])
            .output()
            .unwrap();
        assert!(
            explanation.status.success(),
            "{code} has no registered explanation"
        );
    }
}
