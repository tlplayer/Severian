use severian_driver::{compile_path, run_tests};
use severian_hir::TestMode;
use std::path::{Path, PathBuf};

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/examples")
}

fn severian_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "sev"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn compiles_and_tests_implemented_example_directories() {
    let root = examples_root();
    let directories = [
        "00-getting-started",
        "01-values-control",
        "02-functions-modules",
        "03-collections-iteration",
        "04-classes-traits",
        "05-ownership-borrowing",
        "06-results-patterns",
        "07-generics-constraints",
    ];

    let mut compiled = 0;
    let mut severian_tests = 0;
    for directory in directories {
        for fixture in severian_files(&root.join(directory)) {
            let compilation = compile_path(&fixture)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
            severian_tests += run_tests(&compilation.hir, |_| {})
                .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
            compiled += 1;
        }
    }

    assert_eq!(compiled, 27);
    assert_eq!(severian_tests, 11);
}

#[test]
fn compiles_the_supported_concurrency_examples() {
    let directory = examples_root().join("08-concurrency");
    for fixture in severian_files(&directory).into_iter().take(7) {
        compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
    }
}

#[test]
fn compiles_and_classifies_the_test_gallery() {
    let directory = examples_root().join("15-tests");
    let mut modes = Vec::new();
    let mut tests = 0;

    for fixture in severian_files(&directory) {
        let compilation =
            compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
        tests += run_tests(&compilation.hir, |_| {})
            .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
        for function in &compilation.hir.functions {
            for test in &function.tests {
                modes.extend(test.modes.iter().copied());
            }
        }
    }

    assert_eq!(tests, 7);
    assert!(modes.contains(&TestMode::Property));
    assert!(modes.contains(&TestMode::Bench));
    assert!(modes.contains(&TestMode::Chaos));
}
