use severian_driver::{Compiler, TestExecution};
use severian_mir::{DurationComparison, TestExpectation, TestMode, TestStream};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) fn collect_sources(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            if !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("target" | "errors")
            ) {
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
    let mut skipped = 0usize;
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
            let timeout = test.modes.iter().find_map(|mode| match mode {
                TestMode::Timeout(nanos) => Some(Duration::from_nanos(
                    u64::try_from(*nanos).unwrap_or(u64::MAX),
                )),
                _ => None,
            });
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
            if test
                .modes
                .iter()
                .any(|mode| matches!(mode, TestMode::Model | TestMode::Differential))
            {
                println!(
                    "test {} ... skipped ({} runner is not configured)",
                    test.name,
                    test.modes
                        .iter()
                        .find(|mode| {
                            matches!(mode, TestMode::Model | TestMode::Differential)
                        })
                        .map(|mode| mode.name())
                        .unwrap_or("generated")
                );
                skipped += 1;
                continue;
            }
            if test.modes.contains(&TestMode::Integration) {
                let result = execute(&artifact.path, coverage_file, timeout).map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                let panic = test
                    .expectations
                    .iter()
                    .find_map(|expectation| match expectation {
                        TestExpectation::Panics { function, binding } => {
                            Some((function.as_str(), binding.as_str()))
                        }
                    _ => None,
                });
                let expectation_failure = test.expectations.iter().find_map(|expectation| {
                    let (actual, expected, relation) = match expectation {
                        TestExpectation::Contains { stream, value } => {
                            (test_stream(stream, &result), value, "contain")
                        }
                        TestExpectation::Excludes { stream, value } => {
                            (test_stream(stream, &result), value, "exclude")
                        }
                        TestExpectation::Equals { stream, value } => {
                            (test_stream(stream, &result), value, "equal")
                        }
                        TestExpectation::PanicMessage { binding, value } => {
                            if panic.is_some_and(|(_, panic_binding)| panic_binding == binding) {
                                (String::from_utf8_lossy(&result.stderr), value, "contain")
                            } else {
                                return Some(format!(
                                    "panic message references unknown capture `{binding}`"
                                ));
                            }
                        }
                        TestExpectation::Panics { .. } => return None,
                        TestExpectation::ProfileDuration { .. } => return None,
                        TestExpectation::ProfileMemory { .. } => return None,
                    };
                    let matches = match expectation {
                        TestExpectation::Contains { .. } => actual.contains(expected),
                        TestExpectation::Excludes { .. } => !actual.contains(expected),
                        TestExpectation::Equals { .. } => actual.as_ref() == expected,
                        TestExpectation::PanicMessage { .. } => actual.contains(expected),
                        TestExpectation::Panics { .. } => unreachable!(),
                        TestExpectation::ProfileDuration { .. } => unreachable!(),
                        TestExpectation::ProfileMemory { .. } => unreachable!(),
                    };
                    (!matches).then(|| {
                        format!("captured stream did not {relation} {expected:?}; got {actual:?}")
                    })
                });
                let status_matches = if panic.is_some() {
                    !result.status.success()
                } else {
                    result.status.success()
                };
                if status_matches
                    && expectation_failure.is_none()
                    && !has_soft_expectation_failures(&result)
                {
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
            if test.modes.contains(&TestMode::Benchmark) {
                let warmup = execute(&artifact.path, coverage_file, timeout).map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                if !warmup.status.success() || has_soft_expectation_failures(&warmup) {
                    println!("test {} ... FAILED", test.name);
                    report_captured_output(&warmup);
                    failed += 1;
                    continue;
                }
                let iterations = 10u32;
                let started = Instant::now();
                let mut failure = None;
                for _ in 0..iterations {
                    let result = execute(&artifact.path, coverage_file, timeout).map_err(|error| {
                        format!("could not run {}: {error}", artifact.path.display())
                    })?;
                    if !result.status.success() || has_soft_expectation_failures(&result) {
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
            if test.modes.contains(&TestMode::Profile) {
                let started = Instant::now();
                let result = execute(&artifact.path, coverage_file, timeout).map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                let measured = started.elapsed();
                let timing_failure = test.expectations.iter().find_map(|expectation| {
                    let (comparison, actual, threshold, message, unit) = match expectation {
                        TestExpectation::ProfileDuration {
                            comparison,
                            threshold_nanos,
                            message,
                        } => (comparison, measured.as_nanos(), threshold_nanos, message, "ns"),
                        TestExpectation::ProfileMemory {
                            comparison,
                            threshold_bytes,
                            message,
                        } => (comparison, 0, threshold_bytes, message, "B"),
                        _ => return None,
                    };
                    let satisfied = match comparison {
                        DurationComparison::Less => actual < *threshold,
                        DurationComparison::LessEqual => actual <= *threshold,
                        DurationComparison::Greater => actual > *threshold,
                        DurationComparison::GreaterEqual => actual >= *threshold,
                    };
                    (!satisfied).then(|| {
                        format!("{message}; measured {actual}{unit}, threshold {threshold}{unit}")
                    })
                });
                if result.status.success()
                    && !has_soft_expectation_failures(&result)
                    && timing_failure.is_none()
                {
                    println!("test {} ... profile ({})", test.name, elapsed(measured));
                    passed += 1;
                } else {
                    println!("test {} ... FAILED", test.name);
                    if let Some(message) = timing_failure {
                        eprintln!("{message}");
                    }
                    report_captured_output(&result);
                    failed += 1;
                }
                continue;
            }
            if test
                .modes
                .iter()
                .any(|mode| {
                    !matches!(
                        mode,
                        TestMode::Timeout(_)
                            | TestMode::Property
                            | TestMode::Fuzz
                            | TestMode::Cases
                            | TestMode::Model
                            | TestMode::Differential
                    )
                })
            {
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
            let result = execute(&artifact.path, coverage_file, timeout)
                .map_err(|error| format!("could not run {}: {error}", artifact.path.display()))?;
            if result.status.success() && !has_soft_expectation_failures(&result) {
                if test
                    .modes
                    .iter()
                    .any(|mode| {
                        matches!(
                            mode,
                            TestMode::Property
                                | TestMode::Fuzz
                                | TestMode::Model
                                | TestMode::Differential
                        )
                    })
                {
                    println!("test {} ... ok (seed 0)", test.name);
                } else {
                    println!("test {} ... ok", test.name);
                }
                passed += 1;
            } else {
                println!("test {} ... FAILED", test.name);
                report_captured_output(&result);
                failed += 1;
            }
        }
    }
    println!("\ntest result: {passed} passed; {failed} failed; {skipped} skipped");
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("{failed} test(s) failed"))
    }
}

fn has_soft_expectation_failures(output: &Output) -> bool {
    String::from_utf8_lossy(&output.stderr).contains("expectation failed:")
}

fn execute(
    path: &Path,
    coverage_file: Option<&Path>,
    timeout: Option<Duration>,
) -> std::io::Result<Output> {
    let mut command = Command::new(path);
    if let Some(coverage_file) = coverage_file {
        command.env("SEV_COVERAGE_FILE", coverage_file);
    }
    let Some(timeout) = timeout else {
        return command.output();
    };
    let mut child = command.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output();
        }
        if started.elapsed() >= timeout {
            child.kill()?;
            let mut output = child.wait_with_output()?;
            output.stderr.extend_from_slice(
                format!("test timed out after {}\n", elapsed(timeout)).as_bytes(),
            );
            return Ok(output);
        }
        thread::sleep(Duration::from_millis(1));
    }
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
    format!("{}/iteration", elapsed(value))
}

fn elapsed(value: Duration) -> String {
    if value.as_secs() > 0 {
        format!("{:.3}s", value.as_secs_f64())
    } else if value.as_millis() > 0 {
        format!("{:.3}ms", value.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3}\u{00b5}s", value.as_secs_f64() * 1_000_000.0)
    }
}
