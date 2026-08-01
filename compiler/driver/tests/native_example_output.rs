use severian_driver::{compile_native, compile_path};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/examples")
}

fn severian_files(directory: &Path) -> Vec<PathBuf> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "sev") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    visit(directory, &mut files);
    files.sort();
    files
}

fn has_main(source: &str) -> bool {
    source.lines().any(|line| line.starts_with("def main("))
}

struct TemporaryExecutable(PathBuf);

impl TemporaryExecutable {
    fn new(index: usize) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "severian-native-example-{}-{nonce}-{index}",
            std::process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryExecutable {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn every_example_is_a_verified_native_executable() {
    let root = examples_root();
    let fixtures = severian_files(&root);
    let mut failures = Vec::new();

    for (index, fixture) in fixtures.iter().enumerate() {
        let relative = fixture.strip_prefix(&root).unwrap();
        let source = match std::fs::read_to_string(fixture) {
            Ok(source) => source,
            Err(error) => {
                failures.push(format!(
                    "{}: could not read source: {error}",
                    relative.display()
                ));
                continue;
            }
        };
        if !has_main(&source) {
            failures.push(format!(
                "{}: executable acceptance requires a source main()",
                relative.display()
            ));
        }

        let compilation = match compile_path(fixture) {
            Ok(compilation) => compilation,
            Err(error) => {
                failures.push(format!(
                    "{}: did not reach valid MLIR: {error}",
                    relative.display()
                ));
                continue;
            }
        };
        let executable = TemporaryExecutable::new(index);
        if let Err(error) = compile_native(&compilation, executable.path()) {
            failures.push(format!(
                "{}: native compilation failed: {error}",
                relative.display()
            ));
            continue;
        }

        let output = match Command::new("timeout")
            .arg("5")
            .arg(executable.path())
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!(
                    "{}: could not execute native binary: {error}",
                    relative.display()
                ));
                continue;
            }
        };
        if !output.status.success() {
            failures.push(format!(
                "{}: native executable exited with {}; stderr: {}",
                relative.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        if !output.stderr.is_empty() {
            failures.push(format!(
                "{}: unexpected native stderr: {}",
                relative.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }

        let expected_path = fixture.with_extension("stdout");
        let expected = match std::fs::read(&expected_path) {
            Ok(expected) => expected,
            Err(error) => {
                failures.push(format!(
                    "{}: missing required stdout fixture {}: {error}",
                    relative.display(),
                    expected_path.display()
                ));
                continue;
            }
        };
        if output.stdout != expected {
            failures.push(format!(
                "{}: stdout mismatch; expected {:?}, got {:?}",
                relative.display(),
                String::from_utf8_lossy(&expected),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} mandatory native-acceptance failures across {} examples:\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n")
    );
}
