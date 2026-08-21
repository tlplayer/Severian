use severian_driver::config::{BinaryTarget, Catalog, DeclaredTarget, LibraryTarget, Manifest};
use severian_driver::Compiler;
use severian_target::TargetSpec;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(message) = run(env::args().skip(1).collect()) {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn run(mut arguments: Vec<String>) -> Result<(), String> {
    let catalog = Catalog::load()?;
    if matches!(
        arguments.first().map(String::as_str),
        Some("-h" | "--help" | "help")
    ) {
        print!("{}", help(&catalog));
        return Ok(());
    }
    if matches!(
        arguments.first().map(String::as_str),
        Some("-V" | "--version")
    ) {
        println!("sev {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let command = if arguments
        .first()
        .is_some_and(|argument| is_command(argument))
    {
        arguments.remove(0)
    } else {
        "run".into()
    };
    match command.as_str() {
        "check" => check(parse_common(arguments)?, &catalog),
        "build" | "compile" => build(parse_common(arguments)?, &catalog).map(|_| ()),
        "run" => run_program(parse_common(arguments)?, &catalog),
        "test" => test(parse_common(arguments)?, &catalog),
        "config" => config(arguments, &catalog),
        "new" => create_project(arguments, &catalog, true),
        "init" => create_project(arguments, &catalog, false),
        reserved => Err(format!(
            "`sev {reserved}` is reserved by the stable CLI surface but is not implemented yet"
        )),
    }
}

fn is_command(argument: &str) -> bool {
    matches!(
        argument,
        "check" | "build" | "compile" | "run" | "test" | "new" | "init" | "config"
    )
}

#[derive(Debug, Default)]
struct CommonOptions {
    path: Option<PathBuf>,
    profile: Option<String>,
    backend: Option<String>,
    target: Option<String>,
    bin: Option<String>,
    output: Option<PathBuf>,
    application_args: Vec<String>,
}

fn parse_common(arguments: Vec<String>) -> Result<CommonOptions, String> {
    let mut options = CommonOptions::default();
    let mut cursor = 0;
    while cursor < arguments.len() {
        let argument = &arguments[cursor];
        if argument == "--" {
            options.application_args = arguments[cursor + 1..].to_vec();
            break;
        }
        let destination = match argument.as_str() {
            "--profile" => Some(&mut options.profile),
            "--backend" => Some(&mut options.backend),
            "--target" => Some(&mut options.target),
            "--bin" => Some(&mut options.bin),
            _ => None,
        };
        if let Some(destination) = destination {
            cursor += 1;
            *destination = Some(
                arguments
                    .get(cursor)
                    .ok_or_else(|| format!("{argument} requires a value"))?
                    .clone(),
            );
        } else if matches!(argument.as_str(), "-o" | "--output") {
            cursor += 1;
            options.output = Some(PathBuf::from(
                arguments
                    .get(cursor)
                    .ok_or_else(|| format!("{argument} requires a path"))?,
            ));
        } else if argument.starts_with('-') {
            return Err(format!("unknown option `{argument}`"));
        } else if options.path.is_none() {
            options.path = Some(PathBuf::from(argument));
        } else {
            return Err(format!("unexpected argument `{argument}`"));
        }
        cursor += 1;
    }
    Ok(options)
}

#[derive(Debug, Clone)]
struct ResolvedConfig {
    profile: String,
    backend: String,
    target: String,
    values: BTreeMap<String, ResolvedValue>,
    active_profile: BTreeMap<String, ResolvedValue>,
}

#[derive(Debug, Clone)]
struct ResolvedValue {
    value: String,
    origin: String,
}

fn resolve_config(
    catalog: &Catalog,
    manifest: Option<&Manifest>,
    options: &CommonOptions,
) -> Result<ResolvedConfig, String> {
    let mut values = catalog
        .options
        .iter()
        .map(|option| {
            (
                option.path.clone(),
                ResolvedValue {
                    value: option.default.clone(),
                    origin: "central default".into(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    if let Some(manifest) = manifest {
        for (path, value) in &manifest.values {
            if let Some(resolved) = values.get_mut(path) {
                catalog.validate(path, value)?;
                resolved.value.clone_from(value);
                resolved.origin = "package.toml".into();
            }
        }
    }
    for (path, override_value) in [
        ("build.profile", options.profile.as_ref()),
        ("build.backend", options.backend.as_ref()),
        ("build.target", options.target.as_ref()),
    ] {
        if let Some(value) = override_value {
            catalog.validate(path, value)?;
            let resolved = values
                .get_mut(path)
                .expect("command-line configuration is cataloged");
            resolved.value.clone_from(value);
            resolved.origin = "command line".into();
        }
    }
    let profile = values["build.profile"].value.clone();
    let backend = values["build.backend"].value.clone();
    let target = values["build.target"].value.clone();
    let prefix = format!("profile.{profile}.");
    let active_profile = values
        .iter()
        .filter_map(|(path, resolved)| {
            path.strip_prefix(&prefix).map(|setting| {
                let mut resolved = resolved.clone();
                resolved.origin = format!("{} via {path}", resolved.origin);
                (setting.to_owned(), resolved)
            })
        })
        .collect();
    Ok(ResolvedConfig {
        profile,
        backend,
        target,
        values,
        active_profile,
    })
}

enum Input {
    Source {
        source: PathBuf,
        root: PathBuf,
        name: String,
    },
    Package(Manifest),
}

fn discover(path: Option<&Path>, catalog: &Catalog) -> Result<Input, String> {
    let path = match path {
        Some(path) => path.to_owned(),
        None => env::current_dir()
            .map_err(|error| format!("could not read current directory: {error}"))?,
    };
    if path.is_file() {
        if path.extension().and_then(|extension| extension.to_str()) != Some("sev") {
            return Err(format!("{} is not a Severian source file", path.display()));
        }
        let root = path.parent().unwrap_or_else(|| Path::new(".")).to_owned();
        let name = path
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "app".into());
        return Ok(Input::Source {
            source: path,
            root,
            name,
        });
    }
    if !path.exists() {
        return Err(format!("input `{}` does not exist", path.display()));
    }
    let mut directory = path.as_path();
    loop {
        let manifest = directory.join("package.toml");
        if manifest.is_file() {
            return Manifest::load(&manifest, catalog).map(Input::Package);
        }
        directory = directory.parent().ok_or_else(|| {
            format!(
                "no package.toml found from `{}` to the filesystem root",
                path.display()
            )
        })?;
    }
}

fn check(options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    if !options.application_args.is_empty() {
        return Err("`sev check` does not accept application arguments".into());
    }
    let input = discover(options.path.as_deref(), catalog)?;
    let manifest = match &input {
        Input::Package(manifest) => Some(manifest),
        _ => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let compiler = compiler(&config)?;
    for target in selected_targets(&input, options.bin.as_deref())? {
        compiler
            .check_file(target.path())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn build(options: CommonOptions, catalog: &Catalog) -> Result<Vec<PathBuf>, String> {
    if !options.application_args.is_empty() {
        return Err("`sev build` does not accept application arguments".into());
    }
    let input = discover(options.path.as_deref(), catalog)?;
    let manifest = match &input {
        Input::Package(manifest) => Some(manifest),
        _ => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let compiler = compiler(&config)?;
    let targets = selected_targets(&input, options.bin.as_deref())?;
    if options.output.is_some() && targets.len() != 1 {
        return Err("`--output` requires exactly one selected artifact".into());
    }
    let root = input_root(&input);
    let mut artifacts = Vec::new();
    for target in targets {
        let output = options
            .output
            .clone()
            .unwrap_or_else(|| artifact_path(root, &config, &target));
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        match &target {
            DeclaredTarget::Binary(binary) => {
                compiler
                    .compile_file(&binary.path, &output)
                    .map_err(|error| error.to_string())?;
            }
            DeclaredTarget::Library(library) => {
                compiler
                    .check_file(&library.path)
                    .map_err(|error| error.to_string())?;
                emit_library_package(library, &output)?;
            }
        }
        println!("built {}", output.display());
        artifacts.push(output);
    }
    Ok(artifacts)
}

fn run_program(mut options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    let application_args = std::mem::take(&mut options.application_args);
    let input = discover(options.path.as_deref(), catalog)?;
    let manifest = match &input {
        Input::Package(manifest) => Some(manifest),
        _ => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let binary = selected_binary(&input, options.bin.as_deref())?;
    let output = options.output.clone().unwrap_or_else(|| {
        artifact_path(
            input_root(&input),
            &config,
            &DeclaredTarget::Binary(binary.clone()),
        )
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    compiler(&config)?
        .compile_file(&binary.path, &output)
        .map_err(|error| error.to_string())?;
    let executable = if output.is_absolute() {
        output.clone()
    } else {
        env::current_dir()
            .map_err(|error| format!("could not resolve executable path: {error}"))?
            .join(&output)
    };
    let status = Command::new(&executable)
        .args(application_args)
        .status()
        .map_err(|error| format!("could not run {}: {error}", executable.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("application exited with {status}"))
    }
}

fn test(options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    if !options.application_args.is_empty() {
        return Err("`sev test` does not accept application arguments".into());
    }
    let requested = options.path.clone().unwrap_or_else(|| PathBuf::from("."));
    let (sources, root, manifest) =
        if requested.is_dir() && !requested.join("package.toml").is_file() {
            let mut sources = Vec::new();
            collect_test_sources(&requested, &mut sources)?;
            sources.sort();
            (sources, requested.clone(), None)
        } else {
            let input = discover(Some(&requested), catalog)?;
            let sources = selected_targets(&input, options.bin.as_deref())?
                .into_iter()
                .map(|target| target.path().to_owned())
                .collect();
            let root = input_root(&input).to_owned();
            let manifest = match input {
                Input::Package(manifest) => Some(manifest),
                Input::Source { .. } => None,
            };
            (sources, root, manifest)
        };
    if sources.is_empty() {
        return Err(format!(
            "no Severian test sources found in {}",
            requested.display()
        ));
    }
    let config = resolve_config(catalog, manifest.as_ref(), &options)?;
    let compiler = compiler(&config)?;
    let invocation = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("could not identify test invocation: {error}"))?
        .as_nanos();
    let output_root = root
        .join("target")
        .join(target_directory(&config.target))
        .join(&config.profile)
        .join("tests")
        .join(format!("run-{}-{invocation}", process::id()));
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("could not create {}: {error}", output_root.display()))?;

    let mut passed = 0usize;
    let mut failed = 0usize;
    for (source_index, source) in sources.iter().enumerate() {
        let directory = output_root.join(format!("source-{source_index}"));
        fs::create_dir_all(&directory)
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
        for test in tests {
            let artifact = match &test.execution {
                severian_driver::TestExecution::Compiler { failure } => {
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
                severian_driver::TestExecution::Executable(artifact) => artifact,
            };
            if test.modes == [severian_mir::TestMode::Integration] {
                let result = Command::new(&artifact.path).output().map_err(|error| {
                    format!("could not run {}: {error}", artifact.path.display())
                })?;
                let expectation_failure = test.expectations.iter().find_map(|expectation| {
                    let (actual, expected, relation) = match expectation {
                        severian_mir::TestExpectation::Contains { stream, value } => {
                            (test_stream(stream, &result), value, "contain")
                        }
                        severian_mir::TestExpectation::Equals { stream, value } => {
                            (test_stream(stream, &result), value, "equal")
                        }
                    };
                    let matches = match expectation {
                        severian_mir::TestExpectation::Contains { .. } => actual.contains(expected),
                        severian_mir::TestExpectation::Equals { .. } => actual.as_ref() == expected,
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
            if test.modes == [severian_mir::TestMode::Benchmark] {
                let warmup = Command::new(&artifact.path).output().map_err(|error| {
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
                    let result = Command::new(&artifact.path).output().map_err(|error| {
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
                    let per_iteration = started.elapsed() / iterations;
                    println!("test {} ... bench ({})", test.name, duration(per_iteration));
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
            let result = Command::new(&artifact.path)
                .output()
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

fn test_stream<'a>(
    stream: &severian_mir::TestStream,
    output: &'a std::process::Output,
) -> Cow<'a, str> {
    let bytes = match stream {
        severian_mir::TestStream::Stdout => &output.stdout,
        severian_mir::TestStream::Stderr => &output.stderr,
    };
    String::from_utf8_lossy(bytes)
}

fn report_captured_output(output: &std::process::Output) {
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

fn collect_test_sources(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) != Some("target") {
                collect_test_sources(&path, output)?;
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("sev") {
            output.push(path);
        }
    }
    Ok(())
}

fn compiler(config: &ResolvedConfig) -> Result<Compiler, String> {
    if config.backend == "xla" {
        return Err("whole-program `xla` is not available yet; use `auto` for native programs containing CompileType-selected kernels".into());
    }
    let target = if config.target == "host" {
        TargetSpec::host()
    } else {
        TargetSpec::new(config.target.clone())
    };
    Compiler::new(target).map_err(|error| error.to_string())
}

fn selected_targets(input: &Input, selected: Option<&str>) -> Result<Vec<DeclaredTarget>, String> {
    match input {
        Input::Source { source, name, .. } => {
            if selected.is_some_and(|selected| selected != name) {
                return Err(format!("single source input defines only binary `{name}`"));
            }
            Ok(vec![DeclaredTarget::Binary(BinaryTarget {
                name: name.clone(),
                path: source.clone(),
            })])
        }
        Input::Package(manifest) => {
            if let Some(selected) = selected {
                return manifest
                    .bins
                    .iter()
                    .find(|binary| binary.name == selected)
                    .cloned()
                    .map(|binary| vec![DeclaredTarget::Binary(binary)])
                    .ok_or_else(|| format!("package has no binary named `{selected}`"));
            }
            let mut targets = manifest
                .bins
                .iter()
                .cloned()
                .map(DeclaredTarget::Binary)
                .collect::<Vec<_>>();
            targets.extend(
                manifest
                    .library
                    .iter()
                    .cloned()
                    .map(DeclaredTarget::Library),
            );
            if targets.is_empty() {
                return Err(format!(
                    "package `{}` declares no build targets",
                    manifest.name
                ));
            }
            Ok(targets)
        }
    }
}

fn selected_binary(input: &Input, selected: Option<&str>) -> Result<BinaryTarget, String> {
    match input {
        Input::Source { source, name, .. } => {
            if selected.is_some_and(|selected| selected != name) {
                return Err(format!("single source input defines only binary `{name}`"));
            }
            Ok(BinaryTarget {
                name: name.clone(),
                path: source.clone(),
            })
        }
        Input::Package(manifest) => {
            if manifest.bins.is_empty() {
                return Err(format!(
                    "package `{}` declares no runnable binaries",
                    manifest.name
                ));
            }
            if let Some(selected) = selected {
                return manifest
                    .bins
                    .iter()
                    .find(|binary| binary.name == selected)
                    .cloned()
                    .ok_or_else(|| format!("package has no binary named `{selected}`"));
            }
            if let Some(default) = &manifest.default_run {
                return manifest
                    .bins
                    .iter()
                    .find(|binary| &binary.name == default)
                    .cloned()
                    .ok_or_else(|| format!("default-run names missing binary `{default}`"));
            }
            if manifest.bins.len() == 1 {
                Ok(manifest.bins[0].clone())
            } else {
                Err(
                    "package has multiple binaries; set `package.default-run` or pass `--bin`"
                        .into(),
                )
            }
        }
    }
}

fn input_root(input: &Input) -> &Path {
    match input {
        Input::Source { root, .. } => root,
        Input::Package(manifest) => &manifest.root,
    }
}

fn artifact_path(root: &Path, config: &ResolvedConfig, target: &DeclaredTarget) -> PathBuf {
    let base = root
        .join("target")
        .join(target_directory(&config.target))
        .join(&config.profile);
    match target {
        DeclaredTarget::Binary(binary) => base.join("bin").join(&binary.name),
        DeclaredTarget::Library(library) => base
            .join("pkg")
            .join(format!("{}-{}.pkg", library.name, library.version)),
    }
}

fn emit_library_package(library: &LibraryTarget, output: &Path) -> Result<(), String> {
    let graph = severian_modules::resolve(&library.path).map_err(|error| error.to_string())?;
    let name = library.name.as_bytes();
    let source_root = fs::canonicalize(library.path.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("could not resolve package source root: {error}"))?;
    let mut package = b"SEVPKG\0\x01".to_vec();
    package.extend_from_slice(&(name.len() as u32).to_be_bytes());
    package.extend_from_slice(name);
    package.extend_from_slice(&(graph.modules.len() as u32).to_be_bytes());
    for module in graph.modules {
        let source = fs::read(&module.path)
            .map_err(|error| format!("could not read {}: {error}", module.path.display()))?;
        let relative = module
            .path
            .strip_prefix(&source_root)
            .map_err(|_| {
                format!(
                    "library source import `{}` escapes `{}`",
                    module.path.display(),
                    source_root.display()
                )
            })?
            .to_string_lossy();
        let relative = relative.as_bytes();
        package.extend_from_slice(&(relative.len() as u32).to_be_bytes());
        package.extend_from_slice(relative);
        package.extend_from_slice(&(source.len() as u64).to_be_bytes());
        package.extend_from_slice(&source);
    }
    fs::write(output, package)
        .map_err(|error| format!("could not write {}: {error}", output.display()))
}

fn target_directory(target: &str) -> String {
    target
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn config(arguments: Vec<String>, catalog: &Catalog) -> Result<(), String> {
    let Some(action) = arguments.first().map(String::as_str) else {
        return Err("usage: sev config <show|sync|defaults> [path]".into());
    };
    match action {
        "defaults" => {
            if arguments.len() != 1 {
                return Err("usage: sev config defaults".into());
            }
            print!("{}", catalog.template("hello"));
            Ok(())
        }
        "show" => {
            let options = parse_common(arguments[1..].to_vec())?;
            let input = discover(options.path.as_deref(), catalog)?;
            let manifest = match &input {
                Input::Package(manifest) => Some(manifest),
                _ => None,
            };
            let resolved = resolve_config(catalog, manifest, &options)?;
            for (path, value) in &resolved.values {
                println!("{path} = {:?} # {}", value.value, value.origin);
            }
            for (setting, value) in &resolved.active_profile {
                println!(
                    "active-profile.{setting} = {:?} # {}",
                    value.value, value.origin
                );
            }
            Ok(())
        }
        "sync" => {
            if arguments.len() > 2 {
                return Err("usage: sev config sync [path]".into());
            }
            let path = arguments
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let input = discover(Some(&path), catalog)?;
            let Input::Package(manifest) = input else {
                return Err("config sync requires a package.toml".into());
            };
            let count = catalog.sync(&manifest.root.join("package.toml"))?;
            println!("synchronized {count} configuration option(s)");
            Ok(())
        }
        _ => Err("usage: sev config <show|sync|defaults> [path]".into()),
    }
}

fn create_project(arguments: Vec<String>, catalog: &Catalog, new: bool) -> Result<(), String> {
    if arguments.len() > 1 {
        return Err(if new {
            "usage: sev new <path>"
        } else {
            "usage: sev init [path]"
        }
        .into());
    }
    let root = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if new
        && root.exists()
        && fs::read_dir(&root)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "{} already exists and is not empty",
            root.display()
        ));
    }
    fs::create_dir_all(root.join("src"))
        .map_err(|error| format!("could not create {}: {error}", root.display()))?;
    let name = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty() && name != ".")
        .unwrap_or_else(|| "app".into());
    let manifest = root.join("package.toml");
    if manifest.exists() {
        return Err(format!("{} already exists", manifest.display()));
    }
    fs::write(&manifest, catalog.template(&name))
        .map_err(|error| format!("could not write {}: {error}", manifest.display()))?;
    let source = root.join("src/main.sev");
    if !source.exists() {
        fs::write(&source, "print(\"hello\")\n")
            .map_err(|error| format!("could not write {}: {error}", source.display()))?;
    }
    let lock = root.join("sev.lock");
    if !lock.exists() {
        fs::write(
            &lock,
            format!(
                "# Generated by sev.\nversion = 1\n\n[[package]]\nname = {name:?}\nversion = \"0.1.0\"\n"
            ),
        )
        .map_err(|error| format!("could not write {}: {error}", lock.display()))?;
    }
    println!("initialized {}", root.display());
    Ok(())
}

fn help(catalog: &Catalog) -> String {
    let mut output = String::from(
        "usage: sev [command] [path] [options] [-- application-args]\n\n\
default:\n  sev [path] [-- args...]       Check, build, and run the default binary.\n\n\
commands:\n  check   build   compile   run   test   new   init   config <show|sync|defaults>\n\n\
build options:\n",
    );
    for path in ["build.profile", "build.backend", "build.target"] {
        let option = catalog.get(path).expect("help option is cataloged");
        output.push_str(&format!(
            "  --{:<10} {} (default: {})\n",
            path.trim_start_matches("build."),
            option.description,
            option.default
        ));
    }
    output.push_str("  --bin NAME  Select a package binary.\n  -o PATH     Override the path for one selected artifact.\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_and_backend_are_distinct_options() {
        let options = parse_common(vec![
            "hello.sev".into(),
            "--backend".into(),
            "native".into(),
            "--target".into(),
            "wasm32-unknown-wasi".into(),
            "--".into(),
            "input.txt".into(),
        ])
        .unwrap();
        assert_eq!(options.backend.as_deref(), Some("native"));
        assert_eq!(options.target.as_deref(), Some("wasm32-unknown-wasi"));
        assert_eq!(options.application_args, ["input.txt"]);
    }
}
