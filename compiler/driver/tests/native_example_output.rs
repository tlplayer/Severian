use severian_driver::{compile_native, compile_path};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const VERIFIED_EXAMPLES: &[&str] = &[
    "00-getting-started/01-hello.sev",
    "00-getting-started/02-imports.sev",
    "01-values-control/00-constants.sev",
    "01-values-control/01-bindings.sev",
    "01-values-control/02-if-while-for.sev",
    "01-values-control/03-basic-functions.sev",
    "01-values-control/04-while-initializer.sev",
    "16-compiler-stages/parser-placeholder.sev",
];

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/examples")
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
fn verified_examples_match_their_native_stdout() {
    let root = examples_root();

    for (index, relative_path) in VERIFIED_EXAMPLES.iter().enumerate() {
        let fixture = root.join(relative_path);
        let expected_path = fixture.with_extension("stdout");
        let expected = std::fs::read(&expected_path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", expected_path.display()));
        let compilation =
            compile_path(&fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
        let executable = TemporaryExecutable::new(index);
        compile_native(&compilation, executable.path()).unwrap_or_else(|error| {
            panic!("{}: native compilation failed: {error}", fixture.display())
        });

        let output = Command::new("timeout")
            .arg("5")
            .arg(executable.path())
            .output()
            .unwrap_or_else(|error| {
                panic!("{}: could not run executable: {error}", fixture.display())
            });

        assert!(
            output.status.success(),
            "{}: native executable exited with {}; stderr:\n{}",
            fixture.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stderr.is_empty(),
            "{}: unexpected native stderr:\n{}",
            fixture.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            output.stdout,
            expected,
            "{}: native stdout did not match {}",
            fixture.display(),
            expected_path.display()
        );
    }
}
