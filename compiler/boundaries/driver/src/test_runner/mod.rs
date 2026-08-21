use severian_driver::{Compiler, TestExecution};
use severian_mir::{TestExpectation, TestMode, TestStream};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

pub(crate) fn collect_sources(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) != Some("target") {
                collect_sources(&path, output)?;
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("sev") {
            output.push(path);
        }
    }
    Ok(())
}

pub(crate) fn deduplicate_roots(
    compiler: &Compiler,
    sources: Vec<PathBuf>,
) -> Result<Vec<PathBuf>, String> {
    let mut graphs = sources
        .into_iter()
        .map(|source| {
            let root = fs::canonicalize(&source)
                .map_err(|error| format!("could not resolve {}: {error}", source.display()))?;
            // Discovery must not hide a malformed test behind an early graph
            // error. Keep it as an independent root so the normal compiler
            // path can report the source diagnostic alongside other tests.
            let modules = match compiler.resolved_module_paths(&root) {
                Ok(modules) => modules,
                Err(_) => BTreeSet::from([root.clone()]),
            };
            Ok((root, modules))
        })
        .collect::<Result<Vec<_>, String>>()?;
    graphs.sort_by(|(left_path, left_modules), (right_path, right_modules)| {
        right_modules
            .len()
            .cmp(&left_modules.len())
            .then_with(|| left_path.cmp(right_path))
    });
    let mut covered = BTreeSet::new();
    let mut roots = Vec::new();
    for (root, modules) in graphs {
        if covered.contains(&root) {
            continue;
        }
        covered.extend(modules);
        roots.push(root);
    }
    roots.sort();
    Ok(roots)
}

pub(crate) fn run(
    compiler: &Compiler,
    sources: &[PathBuf],
    output_root: &Path,
) -> Result<(), String> {
    run_with_coverage(compiler, sources, output_root, None)
}

pub(crate) fn run_with_coverage(
    compiler: &Compiler,
    sources: &[PathBuf],
    output_root: &Path,
    coverage_file: Option<&Path>,
) -> Result<(), String> {
    let mut passed = 0usize;
    let mut failed = 0usize;
    for (source_index, source) in sources.iter().enumerate() {
        let directory = output_root.join(format!("source-{source_index}"));
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
        let tests = match compiler.compile_tests_file(source, &directory) {
            Ok(tests) => tests,
            Err(error) => {
                println!("test {} ... FAILED (compile)", source.display());
                eprintln!("{error}");
                failed += 1;
                continue;
            }
        };
        if tests.is_empty() {
            println!("test {} ... ok (compile)", source.display());
            passed += 1;
        }
        for test in tests {
            let artifact = match &test.execution {
                TestExecution::Compiler { failure } => {
                    if let Some(message) = failure {
                        println!("test {} ... FAILED", test.name);
                        eprintln!("{message}");
                        failed += 1;
                    } else {
                        println!("test {} ... ok", test.name);
                        passed += 1;
                    }
                    continue;
                }
                TestExecution::Executable(artifact) => artifact,
            };
            if test.modes == [TestMode::Integration] {
                let result = execute(&artifact.path, coverage_file).map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                let expectation_failure = test.expectations.iter().find_map(|expectation| {
                    let (actual, expected, relation) = match expectation {
                        TestExpectation::Contains { stream, value } => {
                            (test_stream(stream, &result), value, "contain")
                        }
                        TestExpectation::Equals { stream, value } => {
                            (test_stream(stream, &result), value, "equal")
                        }
                    };
                    let matches = match expectation {
                        TestExpectation::Contains { .. } => actual.contains(expected),
                        TestExpectation::Equals { .. } => actual.as_ref() == expected,
                    };
                    (!matches).then(|| {
                        format!("captured stream did not {relation} {expected:?}; got {actual:?}")
                    })
                });
                if result.status.success() && expectation_failure.is_none() {
                    println!("test {} ... ok", test.name);
                    passed += 1;
                } else {
                    println!("test {} ... FAILED", test.name);
                    if let Some(message) = expectation_failure {
                        eprintln!("{message}");
                    }
                    report_captured_output(&result);
                    failed += 1;
                }
                continue;
            }
            if test.modes == [TestMode::Benchmark] {
                let warmup = execute(&artifact.path, coverage_file).map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                if !warmup.status.success() {
                    println!("test {} ... FAILED", test.name);
                    report_captured_output(&warmup);
                    failed += 1;
                    continue;
                }
                let iterations = 10u32;
                let started = Instant::now();
                let mut failure = None;
                for _ in 0..iterations {
                    let result = execute(&artifact.path, coverage_file).map_err(|error| {
                        format!("could not run {}: {error}", artifact.path.display())
                    })?;
                    if !result.status.success() {
                        failure = Some(result);
                        break;
                    }
                }
                if let Some(output) = failure {
                    println!("test {} ... FAILED", test.name);
                    report_captured_output(&output);
                    failed += 1;
                } else {
                    println!(
                        "test {} ... bench ({})",
                        test.name,
                        duration(started.elapsed() / iterations)
                    );
                    passed += 1;
                }
                continue;
            }
            if !test.modes.is_empty() {
                println!(
                    "test {} ... FAILED (unsupported runner: {})",
                    test.name,
                    test.modes
                        .iter()
                        .map(|mode| mode.name())
                        .collect::<Vec<_>>()
                        .join(" and ")
                );
                failed += 1;
                continue;
            }
            let result = execute(&artifact.path, coverage_file)
                .map_err(|error| format!("could not run {}: {error}", artifact.path.display()))?;
            if result.status.success() {
                println!("test {} ... ok", test.name);
                passed += 1;
            } else {
                println!("test {} ... FAILED", test.name);
                report_captured_output(&result);
                failed += 1;
            }
        }
    }
    println!("\ntest result: {passed} passed; {failed} failed; 0 skipped");
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("{failed} test(s) failed"))
    }
}

fn execute(path: &Path, coverage_file: Option<&Path>) -> std::io::Result<Output> {
    let mut command = Command::new(path);
    if let Some(coverage_file) = coverage_file {
        command.env("SEV_COVERAGE_FILE", coverage_file);
    }
    command.output()
}

fn test_stream<'a>(stream: &TestStream, output: &'a Output) -> Cow<'a, str> {
    let bytes = match stream {
        TestStream::Stdout => &output.stdout,
        TestStream::Stderr => &output.stderr,
    };
    String::from_utf8_lossy(bytes)
}

fn report_captured_output(output: &Output) {
    if !output.stdout.is_empty() {
        println!("--- stdout ---");
        print!("{}", String::from_utf8_lossy(&output.stdout));
        if !output.stdout.ends_with(b"\n") {
            println!();
        }
    }
    if !output.stderr.is_empty() {
        let _ = io::stdout().flush();
        eprintln!("--- stderr ---");
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
}

fn duration(value: Duration) -> String {
    if value.as_secs() > 0 {
        format!("{:.3}s/iteration", value.as_secs_f64())
    } else if value.as_millis() > 0 {
        format!("{:.3}ms/iteration", value.as_secs_f64() * 1_000.0)
    } else {
        format!(
            "{:.3}\u{00b5}s/iteration",
            value.as_secs_f64() * 1_000_000.0
        )
    }
}
