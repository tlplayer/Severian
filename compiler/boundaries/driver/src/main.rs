mod example_validation;
mod mutation;
mod test_runner;

use severian_driver::config::{
    registry_release_path, registry_root, BinaryTarget, Catalog, DeclaredTarget, LibraryTarget,
    Manifest,
};
use severian_driver::{Compiler, EmitStage};
use severian_target::TargetSpec;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::time::{SystemTime, UNIX_EPOCH};

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
        "check" => {
            let options = parse_common(arguments)?;
            if options.emit.is_some() {
                emit_ir(options, &catalog)
            } else {
                check(options, &catalog)
            }
        }
        "build" | "compile" => {
            let options = parse_common(arguments)?;
            if options.emit.is_some() {
                emit_ir(options, &catalog)
            } else {
                build(options, &catalog).map(|_| ())
            }
        }
        "run" => {
            let options = parse_common(arguments)?;
            if options.emit.is_some() {
                emit_ir(options, &catalog)
            } else {
                run_program(options, &catalog)
            }
        }
        "test" => {
            let (options, mutate) = parse_test(arguments)?;
            test(options, &catalog, mutate)
        }
        "config" => config(arguments, &catalog),
        "publish" => publish_package(parse_common(arguments)?, &catalog),
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
        "check" | "build" | "compile" | "run" | "test" | "new" | "init" | "config" | "publish"
    )
}

#[derive(Debug, Default)]
struct CommonOptions {
    path: Option<PathBuf>,
    profile: Option<String>,
    target: Option<String>,
    bin: Option<String>,
    output: Option<PathBuf>,
    emit: Option<EmitStage>,
    application_args: Vec<String>,
}

fn parse_emit_stage(value: &str) -> Result<EmitStage, String> {
    match value {
        "ast" => Ok(EmitStage::Ast),
        "hir" => Ok(EmitStage::Hir),
        "mir" => Ok(EmitStage::Mir),
        "lir" => Ok(EmitStage::Lir),
        "mlir" => Ok(EmitStage::Mlir),
        _ => Err(format!(
            "unknown emit stage `{value}`; expected one of: ast, hir, mir, lir, mlir"
        )),
    }
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
        if let Some(value) = argument.strip_prefix("--emit=") {
            options.emit = Some(parse_emit_stage(value)?);
            cursor += 1;
            continue;
        }
        let destination = match argument.as_str() {
            "--profile" => Some(&mut options.profile),
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
        } else if argument == "--emit" {
            cursor += 1;
            let value = arguments
                .get(cursor)
                .ok_or_else(|| "--emit requires a stage".to_owned())?;
            options.emit = Some(parse_emit_stage(value)?);
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

fn parse_test(mut arguments: Vec<String>) -> Result<(CommonOptions, bool), String> {
    let option_end = arguments
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(arguments.len());
    let positions = arguments[..option_end]
        .iter()
        .enumerate()
        .filter_map(|(index, argument)| (argument == "--mutate").then_some(index))
        .collect::<Vec<_>>();
    if positions.len() > 1 {
        return Err("`--mutate` may only be supplied once".into());
    }
    let mutate = if let Some(index) = positions.first().copied() {
        arguments.remove(index);
        true
    } else {
        false
    };
    Ok((parse_common(arguments)?, mutate))
}

fn emit_ir(options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    if !options.application_args.is_empty() {
        return Err("`--emit` does not accept application arguments".into());
    }
    let stage = options.emit.expect("emit dispatch requires a stage");
    let input = discover(options.path.as_deref(), catalog)?;
    let manifest = match &input {
        Input::Package(manifest) => Some(manifest.as_ref()),
        Input::Source { .. } => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let targets = selected_targets(&input, options.bin.as_deref())?;
    if targets.len() != 1 {
        return Err("`--emit` requires exactly one selected target; pass `--bin NAME`".into());
    }
    let text = compiler(&config, manifest, false)?
        .emit_file(targets[0].path(), stage)
        .map_err(|error| error.to_string())?;
    if let Some(output) = &options.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        fs::write(output, text)
            .map_err(|error| format!("could not write {}: {error}", output.display()))?;
    } else {
        print!("{text}");
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ResolvedConfig {
    profile: String,
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
    Package(Box<Manifest>),
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
            return Manifest::load(&manifest, catalog)
                .map(Box::new)
                .map(Input::Package);
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
        Input::Package(manifest) => Some(manifest.as_ref()),
        _ => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let compiler = compiler(&config, manifest, false)?;
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
        Input::Package(manifest) => Some(manifest.as_ref()),
        _ => None,
    };
    let config = resolve_config(catalog, manifest, &options)?;
    let compiler = compiler(&config, manifest, false)?;
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
                emit_library_package(library, &compiler, &output)?;
            }
        }
        println!("built {}", output.display());
        artifacts.push(output);
    }
    Ok(artifacts)
}

fn publish_package(options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    if options.bin.is_some()
        || options.output.is_some()
        || options.emit.is_some()
        || !options.application_args.is_empty()
    {
        return Err("`sev publish` accepts only a package path, profile, and target".into());
    }
    let input = discover(options.path.as_deref(), catalog)?;
    let Input::Package(manifest) = input else {
        return Err("`sev publish` requires a package.toml".into());
    };
    if !manifest.publish {
        return Err(format!(
            "package `{}` is marked publish = false",
            manifest.name
        ));
    }
    let library = manifest
        .library
        .as_ref()
        .ok_or_else(|| format!("package `{}` has no library target", manifest.name))?;
    let config = resolve_config(catalog, Some(&manifest), &options)?;
    let compiler = compiler(&config, Some(&manifest), false)?;
    compiler
        .check_file(&library.path)
        .map_err(|error| error.to_string())?;

    let selected_registry = manifest
        .values
        .get("publish.registry")
        .map(String::as_str)
        .unwrap_or("default");
    let registry = registry_root(Some(selected_registry))?;
    let release = registry_release_path(&registry, &manifest.name, &library.version)?;
    if release.exists() {
        return Err(format!(
            "{} version {} is already published in {}",
            manifest.name,
            library.version,
            registry.display()
        ));
    }
    let invocation = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("could not identify publish invocation: {error}"))?
        .as_nanos();
    let staging = registry.join(format!(".publish-{}-{invocation}", process::id()));
    let source_output = staging.join("source");
    fs::create_dir_all(&source_output)
        .map_err(|error| format!("could not create {}: {error}", source_output.display()))?;

    let source_root = fs::canonicalize(library.path.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("could not resolve package source root: {error}"))?;
    for module in compiler
        .resolved_module_paths(&library.path)
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter(|module| module.starts_with(&source_root))
    {
        let relative = module.strip_prefix(&source_root).map_err(|_| {
            format!(
                "library source import `{}` escapes `{}`",
                module.display(),
                source_root.display()
            )
        })?;
        let destination = source_output.join(relative);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        fs::copy(&module, &destination).map_err(|error| {
            format!(
                "could not copy {} to {}: {error}",
                module.display(),
                destination.display()
            )
        })?;
    }
    let library_relative = library.path.strip_prefix(&source_root).map_err(|_| {
        format!(
            "library source `{}` escapes `{}`",
            library.path.display(),
            source_root.display()
        )
    })?;
    fs::write(
        source_output.join("package.toml"),
        format!(
            "[package]\nname = {:?}\nversion = {:?}\npublish = false\n\n[lib]\nname = {:?}\npath = {:?}\n",
            manifest.name,
            library.version,
            library.name,
            library_relative.to_string_lossy()
        ),
    )
    .map_err(|error| format!("could not write published source manifest: {error}"))?;
    let artifact = staging.join(format!("{}-{}.pkg", manifest.name, library.version));
    emit_library_package(library, &compiler, &artifact)?;
    if let Some(parent) = release.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    fs::rename(&staging, &release).map_err(|error| {
        format!(
            "could not publish {} to {}: {error}",
            manifest.name,
            release.display()
        )
    })?;
    println!(
        "published {} {} to {}",
        manifest.name,
        library.version,
        registry.display()
    );
    Ok(())
}

fn run_program(mut options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    let application_args = std::mem::take(&mut options.application_args);
    let input = discover(options.path.as_deref(), catalog)?;
    let manifest = match &input {
        Input::Package(manifest) => Some(manifest.as_ref()),
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
    compiler(&config, manifest, false)?
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

fn test(options: CommonOptions, catalog: &Catalog, mutate: bool) -> Result<(), String> {
    if options.emit.is_some() {
        return Err("`sev test` does not support `--emit`; emit a selected source directly".into());
    }
    if !options.application_args.is_empty() {
        return Err("`sev test` does not accept application arguments".into());
    }
    let requested = options.path.clone().unwrap_or_else(|| PathBuf::from("."));
    let (mut sources, fixture_packages, root, manifest, validation) =
        if requested.is_dir() && !requested.join("package.toml").is_file() {
            let mut sources = Vec::new();
            test_runner::collect_sources(&requested, &mut sources)?;
            (sources, Vec::new(), requested.clone(), None, None)
        } else {
            let input = discover(Some(&requested), catalog)?;
            let validation = match &input {
                Input::Package(manifest) => manifest.validation.clone(),
                Input::Source { .. } => None,
            };
            let (sources, fixture_packages) = if let Some(validation) = &validation {
                let discovery = example_validation::discover(validation)?;
                (discovery.sources, discovery.packages)
            } else {
                let mut sources = selected_targets(&input, options.bin.as_deref())?
                    .into_iter()
                    .map(|target| target.path().to_owned())
                    .collect::<Vec<_>>();
                if let Input::Package(manifest) = &input {
                    let tests = manifest.root.join("tests");
                    if tests.is_dir() {
                        test_runner::collect_sources(&tests, &mut sources)?;
                    }
                }
                (sources, Vec::new())
            };
            let root = input_root(&input).to_owned();
            let manifest = match input {
                Input::Package(manifest) => Some(*manifest),
                Input::Source { .. } => None,
            };
            (sources, fixture_packages, root, manifest, validation)
        };
    if sources.is_empty() && fixture_packages.is_empty() {
        return Err(format!(
            "no Severian test sources found in {}",
            requested.display()
        ));
    }
    let config = resolve_config(catalog, manifest.as_ref(), &options)?;
    let compiler = compiler(&config, manifest.as_ref(), true)?;
    let compiler = if validation.is_some() {
        compiler.with_coverage()
    } else {
        compiler
    };
    if validation.is_none() {
        sources = test_runner::deduplicate_roots(&compiler, sources)?;
    }
    let invocation = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("could not identify test invocation: {error}"))?
        .as_nanos();
    let output_base = if manifest.is_some() {
        root.join("target")
    } else {
        std::env::temp_dir().join("severian-tests")
    };
    let output_root = output_base
        .join(target_directory(&config.target))
        .join(&config.profile)
        .join("tests")
        .join(format!("run-{}-{invocation}", process::id()));
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("could not create {}: {error}", output_root.display()))?;

    if mutate && validation.is_some() {
        return Err("`sev test --mutate` does not support example-validation packages".into());
    }

    if let (Some(manifest), Some(validation)) = (manifest.as_ref(), validation.as_ref()) {
        example_validation::run(
            &compiler,
            manifest,
            validation,
            example_validation::RunTargets {
                sources: &sources,
                packages: &fixture_packages,
                output_root: &output_root,
            },
            catalog,
            &options,
        )
    } else if mutate {
        mutation::run(&compiler, &sources, &output_root)
    } else {
        test_runner::run(&compiler, &sources, &output_root)
    }
}

fn compiler(
    config: &ResolvedConfig,
    manifest: Option<&Manifest>,
    include_root_dev: bool,
) -> Result<Compiler, String> {
    let target = if config.target == "host" {
        TargetSpec::host()
    } else {
        TargetSpec::new(config.target.clone())
    };
    let max_errors = config.values["diagnostics.max-errors"]
        .value
        .parse::<usize>()
        .map_err(|error| format!("invalid diagnostics.max-errors: {error}"))?;
    Compiler::new(target)
        .map(|compiler| match manifest {
            Some(manifest) => compiler
                .with_max_errors(max_errors)
                .with_packages(manifest.module_graph(include_root_dev)),
            None => compiler.with_max_errors(max_errors),
        })
        .map_err(|error| error.to_string())
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

fn emit_library_package(
    library: &LibraryTarget,
    compiler: &Compiler,
    output: &Path,
) -> Result<(), String> {
    let name = library.name.as_bytes();
    let source_root = fs::canonicalize(library.path.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("could not resolve package source root: {error}"))?;
    let modules = compiler
        .resolved_module_paths(&library.path)
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter(|module| module.starts_with(&source_root))
        .collect::<Vec<_>>();
    let mut package = b"SEVPKG\0\x01".to_vec();
    package.extend_from_slice(&(name.len() as u32).to_be_bytes());
    package.extend_from_slice(name);
    package.extend_from_slice(&(modules.len() as u32).to_be_bytes());
    for module in modules {
        let source = fs::read(&module)
            .map_err(|error| format!("could not read {}: {error}", module.display()))?;
        let relative = module
            .strip_prefix(&source_root)
            .map_err(|_| {
                format!(
                    "library source import `{}` escapes `{}`",
                    module.display(),
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
                Input::Package(manifest) => Some(manifest.as_ref()),
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
commands:\n  check   build   compile   run   test   publish   new   init   config <show|sync|defaults>\n\n\
build options:\n",
    );
    for path in ["build.profile", "build.target"] {
        let option = catalog.get(path).expect("help option is cataloged");
        output.push_str(&format!(
            "  --{:<10} {} (default: {})\n",
            path.trim_start_matches("build."),
            option.description,
            option.default
        ));
    }
    output.push_str(
        "  --bin NAME  Select a package binary.\n  --emit STAGE  Print ast, hir, mir, lir, or mlir; do not execute.\n  -o PATH     Write the selected artifact or emitted IR to PATH.\n\ntest options:\n  --mutate    Run mutation testing.\n",
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_selection_is_separate_from_application_arguments() {
        let options = parse_common(vec![
            "hello.sev".into(),
            "--target".into(),
            "wasm32-unknown-wasi".into(),
            "--".into(),
            "input.txt".into(),
        ])
        .unwrap();
        assert_eq!(options.target.as_deref(), Some("wasm32-unknown-wasi"));
        assert_eq!(options.application_args, ["input.txt"]);
    }

    #[test]
    fn backend_is_resolved_as_a_component_not_a_command_line_option() {
        let error = parse_common(vec!["--backend".into(), "xla".into()]).unwrap_err();
        assert!(error.contains("unknown option `--backend`"));
    }

    #[test]
    fn emit_accepts_separate_and_equals_forms() {
        let separate =
            parse_common(vec!["hello.sev".into(), "--emit".into(), "lir".into()]).unwrap();
        assert_eq!(separate.emit, Some(EmitStage::Lir));

        let equals = parse_common(vec!["hello.sev".into(), "--emit=mlir".into()]).unwrap();
        assert_eq!(equals.emit, Some(EmitStage::Mlir));
    }

    #[test]
    fn mutate_is_a_test_only_flag_in_any_option_position() {
        let (options, mutate) = parse_test(vec![
            "--mutate".into(),
            "fixture.sev".into(),
            "--profile".into(),
            "debug".into(),
        ])
        .unwrap();
        assert!(mutate);
        assert_eq!(options.path.as_deref(), Some(Path::new("fixture.sev")));
        assert_eq!(options.profile.as_deref(), Some("debug"));
        assert!(parse_common(vec!["--mutate".into()]).is_err());
    }
}
