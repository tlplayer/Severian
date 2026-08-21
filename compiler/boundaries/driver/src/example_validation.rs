use crate::test_runner;
use severian_driver::config::{Catalog, Manifest, ValidationManifest};
use severian_driver::Compiler;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Default)]
pub(crate) struct Discovery {
    pub sources: Vec<PathBuf>,
    pub packages: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
struct CoverageSummary {
    lines: f64,
    branches: f64,
}

pub(crate) fn discover(validation: &ValidationManifest) -> Result<Discovery, String> {
    let mut linked = Discovery::default();
    collect_targets(&validation.source, true, &mut linked)?;
    linked.sources.sort();
    linked.packages.sort();
    if linked.sources.is_empty() && linked.packages.is_empty() {
        return Err(format!(
            "no canonical examples found through {}",
            validation.source.display()
        ));
    }

    let mut canonical = Discovery::default();
    collect_targets(&validation.canonical_source, true, &mut canonical)?;
    let discovered_sources = canonical_paths(&linked.sources)?;
    let expected_sources = canonical_paths(&canonical.sources)?;
    let discovered_packages = canonical_paths(&linked.packages)?;
    let expected_packages = canonical_paths(&canonical.packages)?;
    if discovered_sources != expected_sources || discovered_packages != expected_packages {
        let missing = expected_sources.difference(&discovered_sources).count()
            + expected_packages.difference(&discovered_packages).count();
        let orphaned = discovered_sources.difference(&expected_sources).count()
            + discovered_packages.difference(&expected_packages).count();
        return Err(format!(
            "example discovery mismatch: {missing} missing and {orphaned} orphaned target(s)"
        ));
    }
    Ok(linked)
}

pub(crate) fn run(
    compiler: &Compiler,
    manifest: &Manifest,
    validation: &ValidationManifest,
    sources: &[PathBuf],
    fixture_packages: &[PathBuf],
    output_root: &Path,
    catalog: &Catalog,
    options: &super::CommonOptions,
) -> Result<(), String> {
    let coverage_file = output_root.join("coverage.hits");
    let test_result =
        test_runner::run_with_coverage(compiler, sources, output_root, Some(&coverage_file));
    let executable_result = run_executables(
        compiler,
        sources,
        output_root,
        Some(validation),
        Some(&coverage_file),
    );
    let fixture_result = run_package_fixtures(
        fixture_packages,
        output_root,
        catalog,
        options,
        &coverage_file,
    );
    let route_result = observe_routes(compiler, validation, sources);
    let mut coverage_points = declared_coverage(compiler, sources).unwrap_or_default();
    let mut failures = Vec::new();
    for result in [test_result, executable_result] {
        if let Err(error) = result {
            failures.push(error);
        }
    }
    match fixture_result {
        Ok(points) => coverage_points.extend(points),
        Err(error) => failures.push(error),
    }
    let routes = match route_result {
        Ok(routes) => routes,
        Err(error) => {
            failures.push(error);
            BTreeMap::new()
        }
    };
    let coverage = match enforce_coverage(validation, &coverage_points, &coverage_file) {
        Ok(coverage) => coverage,
        Err(error) => {
            failures.push(error);
            CoverageSummary {
                lines: 0.0,
                branches: 0.0,
            }
        }
    };
    if !failures.is_empty() {
        return Err(failures.join("; "));
    }
    write_report(
        manifest,
        validation,
        sources,
        fixture_packages,
        output_root,
        &routes,
        coverage,
    )?;
    println!(
        "validated {} independent source(s) and {} package fixture(s) from {}",
        sources.len(),
        fixture_packages.len(),
        validation.canonical_source.display()
    );
    Ok(())
}

fn observe_routes(
    compiler: &Compiler,
    validation: &ValidationManifest,
    sources: &[PathBuf],
) -> Result<BTreeMap<PathBuf, BTreeSet<String>>, String> {
    let mut observed = BTreeMap::new();
    let mut failures = Vec::new();
    for source in sources {
        let canonical = fs::canonicalize(source)
            .map_err(|error| format!("could not resolve {}: {error}", source.display()))?;
        let routes = match compiler.routes_file(source, true) {
            Ok(routes) => routes,
            Err(_) => continue,
        };
        if let Some(requirement) = validation.examples.get(&canonical) {
            let missing = requirement
                .required_routes
                .difference(&routes)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                failures.push(format!(
                    "{} did not observe required route(s): {}",
                    canonical.display(),
                    missing.join(", ")
                ));
            }
            if !requirement.allow_fallback && routes.iter().any(|route| route.contains("fallback"))
            {
                failures.push(format!(
                    "{} used a forbidden fallback route",
                    canonical.display()
                ));
            }
        }
        observed.insert(canonical, routes);
    }
    for configured in validation.examples.keys() {
        if !observed.contains_key(configured)
            && sources
                .iter()
                .filter_map(|source| fs::canonicalize(source).ok())
                .any(|source| &source == configured)
        {
            failures.push(format!(
                "{} produced no structured route observations",
                configured.display()
            ));
        }
    }
    if failures.is_empty() {
        Ok(observed)
    } else {
        Err(failures.join("; "))
    }
}

fn run_package_fixtures(
    packages: &[PathBuf],
    output_root: &Path,
    catalog: &Catalog,
    options: &super::CommonOptions,
    coverage_file: &Path,
) -> Result<BTreeSet<severian_mir::CoveragePoint>, String> {
    let mut failures = Vec::new();
    let mut coverage_points = BTreeSet::new();
    for (index, package) in packages.iter().enumerate() {
        let result = (|| {
            let manifest = Manifest::load(&package.join("package.toml"), catalog)?;
            let config = super::resolve_config(catalog, Some(&manifest), options)?;
            let compiler = super::compiler(&config, Some(&manifest), true)?.with_coverage();
            let mut sources = manifest
                .bins
                .iter()
                .map(|binary| binary.path.clone())
                .collect::<Vec<_>>();
            sources.extend(manifest.library.iter().map(|library| library.path.clone()));
            let tests = manifest.root.join("tests");
            if tests.is_dir() {
                test_runner::collect_sources(&tests, &mut sources)?;
            }
            let sources = test_runner::deduplicate_roots(&compiler, sources)?;
            coverage_points.extend(declared_coverage(&compiler, &sources)?);
            let package_output = output_root.join(format!("package-{index}"));
            fs::create_dir_all(&package_output).map_err(|error| {
                format!("could not create {}: {error}", package_output.display())
            })?;
            let tests = test_runner::run_with_coverage(
                &compiler,
                &sources,
                &package_output,
                Some(coverage_file),
            );
            let executables = run_executables(
                &compiler,
                &sources,
                &package_output,
                None,
                Some(coverage_file),
            );
            tests.and(executables)
        })();
        match result {
            Ok(()) => println!("package example {} ... ok", package.display()),
            Err(error) => {
                println!("package example {} ... FAILED", package.display());
                eprintln!("{error}");
                failures.push(error);
            }
        }
    }
    if failures.is_empty() {
        Ok(coverage_points)
    } else {
        Err(format!("{} package fixture(s) failed", failures.len()))
    }
}

fn declared_coverage(
    compiler: &Compiler,
    sources: &[PathBuf],
) -> Result<BTreeSet<severian_mir::CoveragePoint>, String> {
    let mut points = BTreeSet::new();
    for source in sources {
        points.extend(
            compiler
                .coverage_points_file(source, true)
                .map_err(|error| error.to_string())?,
        );
        if compiler.file_has_entry(source).unwrap_or(false) {
            points.extend(
                compiler
                    .coverage_points_file(source, false)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    Ok(points)
}

fn enforce_coverage(
    validation: &ValidationManifest,
    points: &BTreeSet<severian_mir::CoveragePoint>,
    coverage_file: &Path,
) -> Result<CoverageSummary, String> {
    let hits = if coverage_file.is_file() {
        fs::read_to_string(coverage_file)
            .map_err(|error| format!("could not read {}: {error}", coverage_file.display()))?
            .lines()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let relevant = points.iter().filter(|point| {
        point
            .file
            .as_deref()
            .map(Path::new)
            .is_some_and(|file| file.starts_with(&validation.canonical_source))
    });
    let mut lines = BTreeMap::<(String, u32), bool>::new();
    let mut branches = BTreeMap::<String, bool>::new();
    for point in relevant {
        let key = point
            .key
            .as_ref()
            .expect("attached coverage point has a key");
        let covered = hits.contains(key);
        match point.kind {
            severian_mir::CoverageKind::Line => {
                let identity = (
                    point.file.clone().expect("coverage point has a file"),
                    point.line.expect("coverage point has a line"),
                );
                lines
                    .entry(identity)
                    .and_modify(|existing| *existing |= covered)
                    .or_insert(covered);
            }
            severian_mir::CoverageKind::Branch => {
                branches.insert(key.clone(), covered);
            }
        }
    }
    let percent = |covered: usize, count: usize| {
        if count == 0 {
            100.0
        } else {
            covered as f64 * 100.0 / count as f64
        }
    };
    let summary = CoverageSummary {
        lines: percent(
            lines.values().filter(|covered| **covered).count(),
            lines.len(),
        ),
        branches: percent(
            branches.values().filter(|covered| **covered).count(),
            branches.len(),
        ),
    };
    let mut failures = Vec::new();
    if summary.lines + f64::EPSILON < f64::from(validation.line_coverage) {
        failures.push(format!(
            "line coverage {:.2}% is below required {}%",
            summary.lines, validation.line_coverage
        ));
    }
    if summary.branches + f64::EPSILON < f64::from(validation.branch_coverage) {
        failures.push(format!(
            "branch coverage {:.2}% is below required {}%",
            summary.branches, validation.branch_coverage
        ));
    }
    if failures.is_empty() {
        Ok(summary)
    } else {
        Err(failures.join("; "))
    }
}

fn collect_targets(directory: &Path, root: bool, output: &mut Discovery) -> Result<(), String> {
    if !root && directory.join("package.toml").is_file() {
        output.packages.push(directory.to_owned());
        return Ok(());
    }
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) != Some("target") {
                collect_targets(&path, false, output)?;
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("sev") {
            output.sources.push(path);
        }
    }
    Ok(())
}

fn canonical_paths(paths: &[PathBuf]) -> Result<BTreeSet<PathBuf>, String> {
    paths
        .iter()
        .map(|path| {
            fs::canonicalize(path)
                .map_err(|error| format!("could not resolve {}: {error}", path.display()))
        })
        .collect()
}

fn run_executables(
    compiler: &Compiler,
    sources: &[PathBuf],
    output_root: &Path,
    validation: Option<&ValidationManifest>,
    coverage_file: Option<&Path>,
) -> Result<(), String> {
    let executable_root = output_root.join("examples");
    fs::create_dir_all(&executable_root).map_err(|error| {
        format!(
            "could not create example output directory {}: {error}",
            executable_root.display()
        )
    })?;
    let mut failures = 0usize;
    for (index, source) in sources.iter().enumerate() {
        let has_entry = match compiler.file_has_entry(source) {
            Ok(has_entry) => has_entry,
            Err(_) => continue,
        };
        if !has_entry {
            continue;
        }
        let canonical = fs::canonicalize(source)
            .map_err(|error| format!("could not resolve {}: {error}", source.display()))?;
        let requirement = validation.and_then(|validation| validation.examples.get(&canonical));
        let status_fixture = source.with_extension("status");
        let configured_exit = requirement.and_then(|requirement| requirement.expected_exit);
        let fixture_exit = if status_fixture.is_file() {
            let value = fs::read_to_string(&status_fixture)
                .map_err(|error| format!("could not read {}: {error}", status_fixture.display()))?;
            Some(value.trim().parse::<i32>().map_err(|_| {
                format!(
                    "{} must contain an integer exit code",
                    status_fixture.display()
                )
            })?)
        } else {
            None
        };
        if configured_exit.is_some() && fixture_exit.is_some() && configured_exit != fixture_exit {
            return Err(format!(
                "{} has conflicting configured and adjacent exit expectations",
                source.display()
            ));
        }
        let expected_exit = configured_exit.or(fixture_exit).unwrap_or(0);
        let has_output_fixture =
            source.with_extension("stdout").is_file() || source.with_extension("stderr").is_file();
        let has_assertions = compiler.file_has_asserting_tests(source).unwrap_or(false);
        if !has_output_fixture
            && configured_exit.is_none()
            && fixture_exit.is_none()
            && !has_assertions
        {
            println!(
                "example {} ... FAILED (missing expectation)",
                source.display()
            );
            eprintln!(
                "runnable example {} requires asserting tests or exact stdout/stderr/exit expectations",
                canonical.display()
            );
            failures += 1;
            continue;
        }
        let executable = executable_root.join(format!("example-{index}"));
        let artifact = match compiler.compile_file(source, &executable) {
            Ok(artifact) => artifact,
            Err(error) => {
                println!("example {} ... FAILED (compile)", source.display());
                eprintln!("{error}");
                failures += 1;
                continue;
            }
        };
        let mut command = Command::new(&artifact.path);
        if let Some(coverage_file) = coverage_file {
            command.env("SEV_COVERAGE_FILE", coverage_file);
        }
        let output = command
            .output()
            .map_err(|error| format!("could not run {}: {error}", artifact.path.display()))?;
        let expectation = compare_fixture(source, "stdout", &output.stdout)
            .and_then(|()| compare_fixture(source, "stderr", &output.stderr));
        if output.status.code() == Some(expected_exit) && expectation.is_ok() {
            println!("example {} ... ok", source.display());
        } else {
            println!("example {} ... FAILED", source.display());
            if let Err(message) = expectation {
                eprintln!("{message}");
            }
            if output.status.code() != Some(expected_exit) {
                eprintln!(
                    "expected exit code {expected_exit}, got {:?}",
                    output.status.code()
                );
            }
            if !output.stdout.is_empty() {
                println!(
                    "--- stdout ---\n{}",
                    String::from_utf8_lossy(&output.stdout)
                );
            }
            if !output.stderr.is_empty() {
                eprintln!(
                    "--- stderr ---\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            failures += 1;
        }
    }
    if failures == 0 {
        Ok(())
    } else {
        Err(format!("{failures} executable example(s) failed"))
    }
}

fn compare_fixture(source: &Path, extension: &str, actual: &[u8]) -> Result<(), String> {
    let fixture = source.with_extension(extension);
    if !fixture.is_file() {
        return Ok(());
    }
    let expected = fs::read(&fixture)
        .map_err(|error| format!("could not read {}: {error}", fixture.display()))?;
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{} did not match {}", extension, fixture.display()))
    }
}

fn write_report(
    manifest: &Manifest,
    validation: &ValidationManifest,
    sources: &[PathBuf],
    fixture_packages: &[PathBuf],
    output_root: &Path,
    routes: &BTreeMap<PathBuf, BTreeSet<String>>,
    coverage: CoverageSummary,
) -> Result<(), String> {
    let root_package = manifest
        .package_graph
        .packages
        .get(&manifest.package_graph.root)
        .expect("validation package graph contains its root");
    let dependencies = root_package
        .dependencies
        .keys()
        .chain(root_package.dev_dependencies.keys())
        .map(|name| format!("\"{}\"", json(name)))
        .collect::<Vec<_>>()
        .join(", ");
    let examples = sources
        .iter()
        .map(|source| {
            fs::canonicalize(source)
                .map(|source| {
                    let observed_routes = routes
                        .get(&source)
                        .into_iter()
                        .flat_map(|routes| routes.iter())
                        .map(|route| format!("\"{}\"", json(route)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "    {{\"source\": \"{}\", \"module\": \"{}\", \"routes\": [{}]}}",
                        json(&source.display().to_string()),
                        json(&module_identity(&validation.canonical_source, &source)),
                        observed_routes,
                    )
                })
                .map_err(|error| format!("could not resolve {}: {error}", source.display()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(",\n");
    let fixtures = fixture_packages
        .iter()
        .map(|package| {
            fs::canonicalize(package)
                .map(|package| format!("\"{}\"", json(&package.display().to_string())))
                .map_err(|error| format!("could not resolve {}: {error}", package.display()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let report = format!(
        "{{\n  \"package\": \"{}\",\n  \"canonical_source\": \"{}\",\n  \"dependencies\": [{}],\n  \"coverage\": {{\"lines\": {:.2}, \"branches\": {:.2}, \"required_lines\": {}, \"required_branches\": {}}},\n  \"examples\": [\n{}\n  ],\n  \"package_fixtures\": [{}]\n}}\n",
        json(&manifest.name),
        json(&validation.canonical_source.display().to_string()),
        dependencies,
        coverage.lines,
        coverage.branches,
        validation.line_coverage,
        validation.branch_coverage,
        examples,
        fixtures,
    );
    let path = output_root.join("examples-validation.json");
    fs::write(&path, report)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    println!("validation report {}", path.display());
    Ok(())
}

fn module_identity(root: &Path, source: &Path) -> String {
    source
        .strip_prefix(root)
        .unwrap_or(source)
        .with_extension("")
        .display()
        .to_string()
}

fn json(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
