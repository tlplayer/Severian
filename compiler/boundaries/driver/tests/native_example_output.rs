use severian_driver::{compile_native, compile_native_tests, compile_path};
use severian_xla::{PjrtPlugin, XlaError};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
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

struct NativeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    truncated: bool,
}

fn run_native(executable: &Path) -> std::io::Result<NativeOutput> {
    const OUTPUT_LIMIT: u64 = 1024 * 1024;
    let mut child = Command::new("timeout")
        .arg("5")
        .arg(executable)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(OUTPUT_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr
            .take(OUTPUT_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let status = child.wait()?;
    let mut stdout = stdout_reader
        .join()
        .expect("stdout reader must not panic")?;
    let mut stderr = stderr_reader
        .join()
        .expect("stderr reader must not panic")?;
    let truncated = stdout.len() > OUTPUT_LIMIT as usize || stderr.len() > OUTPUT_LIMIT as usize;
    stdout.truncate(OUTPUT_LIMIT as usize);
    stderr.truncate(OUTPUT_LIMIT as usize);
    Ok(NativeOutput {
        status,
        stdout,
        stderr,
        truncated,
    })
}

#[test]
fn every_example_is_a_verified_native_executable() {
    let root = examples_root();
    // Small-model inference is a separate post-acceptance gate. It must not be
    // treated as green until the language/examples corpus passes on its own.
    let fixtures = severian_files(&root)
        .into_iter()
        .filter(|fixture| {
            !fixture
                .strip_prefix(&root)
                .is_ok_and(|relative| relative.starts_with("30-llm"))
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    let pjrt_available = match PjrtPlugin::load_rocm() {
        Ok(_) => true,
        Err(XlaError::PluginLoad(message))
            if std::env::var_os("SEVERIAN_ROCM_PJRT_PLUGIN").is_none()
                && message.contains("not found") =>
        {
            false
        }
        Err(error) => panic!("configured PJRT plugin is unusable: {error}"),
    };
    let loopback_available = std::net::TcpListener::bind(("127.0.0.1", 0)).is_ok();

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
        if fixture
            .file_name()
            .is_some_and(|name| name == "invalid.sev")
        {
            let expected_path = fixture.parent().unwrap().join("expected-error.txt");
            let expected = match std::fs::read_to_string(&expected_path) {
                Ok(expected) => expected,
                Err(error) => {
                    failures.push(format!(
                        "{}: missing expected diagnostic {}: {error}",
                        relative.display(),
                        expected_path.display()
                    ));
                    continue;
                }
            };
            match compile_path(fixture) {
                Ok(_) => failures.push(format!(
                    "{}: expected-invalid source compiled successfully",
                    relative.display()
                )),
                Err(error) => {
                    let actual = error.to_string();
                    for required in expected
                        .lines()
                        .take(1)
                        .chain(expected.lines().skip(3).take(1))
                    {
                        if !actual.contains(required) {
                            failures.push(format!(
                                "{}: diagnostic {:?} did not contain {:?}",
                                relative.display(),
                                actual,
                                required
                            ));
                        }
                    }
                }
            }
            continue;
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
        let native_result = if has_main(&source) {
            compile_native(&compilation, executable.path())
        } else {
            compile_native_tests(&compilation, executable.path()).map(|_| ())
        };
        if let Err(error) = native_result {
            failures.push(format!(
                "{}: native compilation failed: {error}",
                relative.display()
            ));
            continue;
        }
        let requires_pjrt = compilation.optimized_hir.functions.iter().any(|function| {
            function
                .decorators
                .iter()
                .any(|decorator| decorator.package == "tensor")
        });
        if requires_pjrt && !pjrt_available {
            eprintln!(
                "skipping optional PJRT execution for {}; native compilation succeeded",
                relative.display()
            );
            continue;
        }
        let requires_loopback = source.contains("database.start(");
        if requires_loopback && !loopback_available {
            eprintln!(
                "skipping loopback-restricted execution for {}; native compilation succeeded",
                relative.display()
            );
            continue;
        }

        let output = match run_native(executable.path()) {
            Ok(output) => output,
            Err(error) => {
                failures.push(format!(
                    "{}: could not execute native binary: {error}",
                    relative.display()
                ));
                continue;
            }
        };
        if output.truncated {
            failures.push(format!(
                "{}: native output exceeded the 1 MiB safety limit",
                relative.display()
            ));
        }
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
