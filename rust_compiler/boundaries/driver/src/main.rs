mod api;
mod example_validation;
mod mutation;
mod test_runner;

use severian_driver::config::{
    registry_release_path, registry_root, BinaryTarget, Catalog, DeclaredTarget, LibraryTarget,
    Manifest,
};
use severian_driver::{Compiler, EmitStage};
use severian_target::TargetSpec;
use std::collections::{BTreeMap, BTreeSet};
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
        "doctor" => doctor(arguments),
        "api" => api::run(arguments),
        "config" => config(arguments, &catalog),
        "publish" => publish_package(parse_common(arguments)?, &catalog),
        "add" => edit_dependency(DependencyEdit::Add, arguments, &catalog),
        "remove" => edit_dependency(DependencyEdit::Remove, arguments, &catalog),
        "update" => edit_dependency(DependencyEdit::Update, arguments, &catalog),
        "install" => install_package(arguments, &catalog),
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
        "check"
            | "build"
            | "compile"
            | "run"
            | "test"
            | "doctor"
            | "api"
            | "new"
            | "init"
            | "config"
            | "publish"
            | "add"
            | "remove"
            | "update"
            | "install"
    )
}

fn doctor(arguments: Vec<String>) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("usage: sev doctor".into());
    }
    let target = TargetSpec::host();
    println!("Severian        {}", env!("CARGO_PKG_VERSION"));
    println!("Target          {}", target.triple);
    println!(
        "Native compiler {}",
        availability(
            tool_available("SEVERIAN_CLANG", "clang-21")
                && tool_available("SEVERIAN_LINKER", "ld.lld-21")
        )
    );
    println!(
        "LLVM backend    {}",
        availability(
            tool_available("SEVERIAN_MLIR_OPT", "mlir-opt-21")
                && tool_available("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21")
                && tool_available("SEVERIAN_LLVM_CONFIG", "llvm-config-21")
        )
    );
    println!("SIMD backend    available");
    println!(
        "ROCm backend    {}",
        availability(target.rocm_device().is_some())
    );
    println!("CUDA backend    not found");
    println!(
        "XLA backend     {}",
        availability(tool_available("SEVERIAN_XLA", "xla"))
    );
    Ok(())
}

fn availability(available: bool) -> &'static str {
    if available {
        "available"
    } else {
        "not found"
    }
}

fn tool_available(variable: &str, default: &str) -> bool {
    let program = env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    Command::new(program)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
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
        "agent-ir" => Ok(EmitStage::AgentIr),
        _ => Err(format!(
            "unknown emit stage `{value}`; expected one of: ast, hir, mir, lir, mlir, agent-ir"
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
    if stage == EmitStage::AgentIr {
        let package = match &input {
            Input::Source { name, .. } => name.as_str(),
            Input::Package(manifest) => manifest.name.as_str(),
        };
        let output = options
            .output
            .clone()
            .unwrap_or_else(|| input_root(&input).join("target").join("agent-ir"));
        compiler(&config, manifest, false)?
            .emit_agent_ir(targets[0].path(), input_root(&input), &output, package)
            .map_err(|error| error.to_string())?;
        println!("wrote Agent IR to {}", output.display());
        return Ok(());
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
    if manifest.library.is_none() && manifest.bins.is_empty() {
        return Err(format!(
            "package `{}` has no publishable target",
            manifest.name
        ));
    }
    let config = resolve_config(catalog, Some(&manifest), &options)?;
    let compiler = compiler(&config, Some(&manifest), false)?;
    for source in manifest
        .library
        .iter()
        .map(|library| &library.path)
        .chain(manifest.bins.iter().map(|binary| &binary.path))
    {
        compiler
            .check_file(source)
            .map_err(|error| error.to_string())?;
    }

    let selected_registry = manifest
        .values
        .get("publish.registry")
        .map(String::as_str)
        .unwrap_or("default");
    let registry = registry_root(Some(selected_registry))?;
    let release = registry_release_path(&registry, &manifest.name, &manifest.version)?;
    if release.exists() {
        return Err(format!(
            "{} version {} is already published in {}",
            manifest.name,
            manifest.version,
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

    let source_root = fs::canonicalize(&manifest.root)
        .map_err(|error| format!("could not resolve package source root: {error}"))?;
    let mut modules = BTreeSet::new();
    for source in manifest
        .library
        .iter()
        .map(|library| &library.path)
        .chain(manifest.bins.iter().map(|binary| &binary.path))
    {
        modules.extend(
            compiler
                .resolved_module_paths(source)
                .map_err(|error| error.to_string())?
                .into_iter()
                .filter(|module| module.starts_with(&source_root)),
        );
    }
    for module in modules {
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
    let published_manifest = manifest.published_source_manifest()?;
    fs::write(source_output.join("package.toml"), &published_manifest)
        .map_err(|error| format!("could not write published source manifest: {error}"))?;
    let metadata = staging.join("metadata");
    fs::create_dir_all(&metadata)
        .map_err(|error| format!("could not create {}: {error}", metadata.display()))?;
    fs::write(metadata.join("package.toml"), &published_manifest)
        .map_err(|error| format!("could not write published package metadata: {error}"))?;
    fs::write(metadata.join("sev.lock"), manifest.render_lockfile()?)
        .map_err(|error| format!("could not write published dependency lock: {error}"))?;
    fs::write(
        metadata.join("build.toml"),
        format!(
            "format = 1\ncompiler = {:?}\ntarget = {:?}\nprofile = {:?}\n",
            env!("CARGO_PKG_VERSION"),
            target_directory(&config.target),
            config.profile
        ),
    )
    .map_err(|error| format!("could not write published build metadata: {error}"))?;

    let binary_artifact_root = staging
        .join("artifacts")
        .join(target_directory(&config.target))
        .join(&config.profile)
        .join("bin");
    let mut artifact_index = String::from("version = 1\n");
    for binary in &manifest.bins {
        fs::create_dir_all(&binary_artifact_root).map_err(|error| {
            format!(
                "could not create {}: {error}",
                binary_artifact_root.display()
            )
        })?;
        let output = binary_artifact_root.join(&binary.name);
        compiler
            .compile_file(&binary.path, &output)
            .map_err(|error| error.to_string())?;
        artifact_index.push_str(&format!(
            "\n[[artifact]]\nkind = \"binary\"\nname = {:?}\ntarget = {:?}\nprofile = {:?}\npath = {:?}\n",
            binary.name,
            target_directory(&config.target),
            config.profile,
            format!(
                "artifacts/{}/{}/bin/{}",
                target_directory(&config.target),
                config.profile,
                binary.name
            )
        ));
    }
    fs::write(metadata.join("artifacts.toml"), artifact_index)
        .map_err(|error| format!("could not write artifact index: {error}"))?;
    let artifact = staging.join(format!("{}-{}.pkg", manifest.name, manifest.version));
    emit_distribution_package(&staging, &artifact)?;
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
        manifest.version,
        registry.display()
    );
    Ok(())
}

fn run_program(mut options: CommonOptions, catalog: &Catalog) -> Result<(), String> {
    let application_args = std::mem::take(&mut options.application_args);
    if let Some(specification) = options.path.as_ref().and_then(|path| {
        let value = path.to_string_lossy();
        (!path.exists() && is_git_specification(&value)).then(|| value.into_owned())
    }) {
        options.path = Some(materialize_git_package(&specification)?);
    }
    if let Some(specification) = options.path.as_ref().and_then(|path| {
        (!path.exists())
            .then(|| path.to_string_lossy().into_owned())
            .filter(|value| !value.contains('/') && !value.contains('\\'))
    }) {
        let (executable, _) = materialize_registry_binary(&specification, &options, catalog)?;
        return execute_binary(&executable, &application_args);
    }
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
    execute_binary(&executable, &application_args)
}

fn is_git_specification(value: &str) -> bool {
    value.starts_with("github.com/")
        || value.starts_with("https://github.com/")
        || value.starts_with("http://github.com/")
        || value.starts_with("git+")
}

fn materialize_git_package(specification: &str) -> Result<PathBuf, String> {
    let raw = specification.strip_prefix("git+").unwrap_or(specification);
    let (location, revision) = raw
        .split_once('#')
        .map_or((raw, None), |(location, revision)| {
            (location, Some(revision))
        });
    let location = if location.starts_with("github.com/") {
        format!("https://{location}")
    } else {
        location.to_owned()
    };
    let registry = registry_root(None)?;
    let identity = format!("{location}#{}", revision.unwrap_or("HEAD"));
    let checkout = registry
        .join("cache/git")
        .join(format!("{:016x}", fnv1a64(identity.as_bytes())));
    if checkout.join("package.toml").is_file() {
        return Ok(checkout);
    }
    let parent = checkout
        .parent()
        .ok_or_else(|| "invalid Git package cache path".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(
        ".clone-{}-{}",
        process::id(),
        fnv1a64(identity.as_bytes())
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)
            .map_err(|error| format!("could not clear stale {}: {error}", temporary.display()))?;
    }
    let clone = Command::new("git")
        .args(["clone", "--depth", "1", "--", &location])
        .arg(&temporary)
        .output()
        .map_err(|error| format!("could not launch git for `{location}`: {error}"))?;
    if !clone.status.success() {
        let _ = fs::remove_dir_all(&temporary);
        return Err(format!(
            "could not resolve Git package `{specification}`: {}",
            String::from_utf8_lossy(&clone.stderr).trim()
        ));
    }
    if let Some(revision) = revision {
        let checkout_result = Command::new("git")
            .args(["-C"])
            .arg(&temporary)
            .args(["checkout", "--detach", revision])
            .output()
            .map_err(|error| format!("could not select Git revision `{revision}`: {error}"))?;
        if !checkout_result.status.success() {
            let _ = fs::remove_dir_all(&temporary);
            return Err(format!(
                "could not select Git revision `{revision}`: {}",
                String::from_utf8_lossy(&checkout_result.stderr).trim()
            ));
        }
    }
    if !temporary.join("package.toml").is_file() {
        let _ = fs::remove_dir_all(&temporary);
        return Err(format!(
            "Git package `{specification}` has no root package.toml"
        ));
    }
    fs::rename(&temporary, &checkout)
        .map_err(|error| format!("could not commit Git package cache: {error}"))?;
    Ok(checkout)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn execute_binary(executable: &Path, arguments: &[String]) -> Result<(), String> {
    let status = Command::new(executable)
        .args(arguments)
        .status()
        .map_err(|error| format!("could not run {}: {error}", executable.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("application exited with {status}"))
    }
}

fn materialize_registry_binary(
    specification: &str,
    options: &CommonOptions,
    catalog: &Catalog,
) -> Result<(PathBuf, String), String> {
    let (package, requirement) = package_specification(specification)?;
    let version = select_registry_version(&package, requirement.as_deref(), None)?;
    let registry = registry_root(None)?;
    let release = registry_release_path(&registry, &package, &version.spelling)?;
    let distribution = ensure_distribution_root(&registry, &release, &package, &version.spelling)?;
    let metadata_path = distribution.join("metadata/package.toml");
    let source_path = distribution.join("source/package.toml");
    let manifest_text = fs::read_to_string(&metadata_path)
        .or_else(|_| fs::read_to_string(&source_path))
        .map_err(|error| {
            format!(
                "package `{package}` {} has no readable distribution metadata: {error}",
                version.spelling
            )
        })?;
    let metadata = manifest_text
        .parse::<toml::Value>()
        .map_err(|error| format!("invalid {}: {error}", metadata_path.display()))?;
    let binary_name = distribution_binary_name(&metadata, options.bin.as_deref())?;
    let target = options
        .target
        .clone()
        .or_else(|| manifest_setting(&metadata, "build", "target"))
        .unwrap_or_else(|| "host".into());
    let profile = options
        .profile
        .clone()
        .or_else(|| manifest_setting(&metadata, "build", "profile"))
        .unwrap_or_else(|| "dev".into());
    let artifact = distribution
        .join("artifacts")
        .join(target_directory(&target))
        .join(&profile)
        .join("bin")
        .join(&binary_name);
    if artifact.is_file() {
        return Ok((artifact, binary_name));
    }
    if !source_path.is_file() {
        return Err(format!(
            "package `{package}` {} has no compatible `{}`/`{profile}` binary `{binary_name}` and no source fallback",
            version.spelling,
            target_directory(&target)
        ));
    }
    let manifest = Manifest::load(&source_path, catalog)?;
    let mut source_options = CommonOptions::default();
    source_options.profile = Some(profile.clone());
    source_options.target = Some(target.clone());
    source_options.bin = Some(binary_name.clone());
    let config = resolve_config(catalog, Some(&manifest), &source_options)?;
    let input = Input::Package(Box::new(manifest.clone()));
    let binary = selected_binary(&input, Some(&binary_name))?;
    let cache = registry
        .join("cache")
        .join("packages")
        .join(&package)
        .join(&version.spelling)
        .join(target_directory(&config.target))
        .join(&config.profile)
        .join("bin")
        .join(&binary_name);
    if let Some(parent) = cache.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    compiler(&config, Some(&manifest), false)?
        .compile_file(&binary.path, &cache)
        .map_err(|error| error.to_string())?;
    Ok((cache, binary_name))
}

fn ensure_distribution_root(
    registry: &Path,
    release: &Path,
    package: &str,
    version: &str,
) -> Result<PathBuf, String> {
    if release.join("metadata/package.toml").is_file() {
        return Ok(release.to_owned());
    }
    let archive = release.join(format!("{package}-{version}.pkg"));
    if !archive.is_file() {
        return Ok(release.to_owned());
    }
    let cache = registry
        .join("cache/distributions")
        .join(package)
        .join(version);
    if cache.join("metadata/package.toml").is_file() {
        return Ok(cache);
    }
    let parent = cache
        .parent()
        .ok_or_else(|| "invalid distribution cache path".to_owned())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let staging = parent.join(format!(".extract-{}-{version}", process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .map_err(|error| format!("could not clear stale {}: {error}", staging.display()))?;
    }
    fs::create_dir_all(&staging)
        .map_err(|error| format!("could not create {}: {error}", staging.display()))?;
    if let Err(error) = unpack_distribution_package(&archive, &staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if !staging.join("metadata/package.toml").is_file() {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!(
            "distribution `{}` has no metadata/package.toml",
            archive.display()
        ));
    }
    match fs::rename(&staging, &cache) {
        Ok(()) => Ok(cache),
        Err(_) if cache.join("metadata/package.toml").is_file() => {
            let _ = fs::remove_dir_all(&staging);
            Ok(cache)
        }
        Err(error) => Err(format!(
            "could not commit extracted distribution {}: {error}",
            cache.display()
        )),
    }
}

fn distribution_binary_name(
    manifest: &toml::Value,
    requested: Option<&str>,
) -> Result<String, String> {
    let binaries = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|binary| binary.get("name").and_then(toml::Value::as_str))
        .collect::<Vec<_>>();
    if let Some(requested) = requested {
        return binaries
            .contains(&requested)
            .then(|| requested.to_owned())
            .ok_or_else(|| format!("package has no binary named `{requested}`"));
    }
    if let Some(default) = manifest
        .get("package")
        .and_then(|package| package.get("default-run"))
        .and_then(toml::Value::as_str)
    {
        return binaries
            .contains(&default)
            .then(|| default.to_owned())
            .ok_or_else(|| format!("package.default-run names missing binary `{default}`"));
    }
    match binaries.as_slice() {
        [binary] => Ok((*binary).to_owned()),
        [] => Err("package does not expose an executable binary".into()),
        _ => Err("package has multiple binaries; select one with `--bin NAME`".into()),
    }
}

fn manifest_setting(manifest: &toml::Value, table: &str, key: &str) -> Option<String> {
    manifest
        .get(table)
        .and_then(|value| value.get(key))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
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
    } else if config.target == "gpu" {
        let mut target = TargetSpec::host();
        target.capabilities.insert("execution.default.gpu");
        target
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

fn emit_distribution_package(root: &Path, output: &Path) -> Result<(), String> {
    fn collect(root: &Path, directory: &Path, entries: &mut Vec<PathBuf>) -> Result<(), String> {
        let mut children = fs::read_dir(directory)
            .map_err(|error| format!("could not read {}: {error}", directory.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("could not enumerate {}: {error}", directory.display()))?;
        children.sort_by_key(std::fs::DirEntry::file_name);
        for child in children {
            let path = child.path();
            let kind = child
                .file_type()
                .map_err(|error| format!("could not inspect {}: {error}", path.display()))?;
            if kind.is_symlink() {
                return Err(format!(
                    "package distribution entry `{}` may not be a symlink",
                    path.display()
                ));
            }
            if kind.is_dir() {
                collect(root, &path, entries)?;
            } else if kind.is_file() {
                path.strip_prefix(root).map_err(|_| {
                    format!(
                        "package entry `{}` escapes its staging root",
                        path.display()
                    )
                })?;
                entries.push(path);
            }
        }
        Ok(())
    }

    let mut entries = Vec::new();
    collect(root, root, &mut entries)?;
    entries.sort_by(|left, right| {
        left.strip_prefix(root)
            .unwrap()
            .cmp(right.strip_prefix(root).unwrap())
    });
    let mut package = b"SEVPKG\0\x02".to_vec();
    package.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for entry in entries {
        let relative = entry
            .strip_prefix(root)
            .map_err(|_| format!("package entry `{}` escapes its root", entry.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        if relative.starts_with('/') || relative.split('/').any(|component| component == "..") {
            return Err(format!("unsafe package entry `{relative}`"));
        }
        let contents = fs::read(&entry)
            .map_err(|error| format!("could not read {}: {error}", entry.display()))?;
        package.extend_from_slice(&(relative.len() as u32).to_be_bytes());
        package.extend_from_slice(relative.as_bytes());
        package.push(u8::from(is_executable(&entry)?));
        package.extend_from_slice(&(contents.len() as u64).to_be_bytes());
        package.extend_from_slice(&contents);
    }
    fs::write(output, package)
        .map_err(|error| format!("could not write {}: {error}", output.display()))
}

fn is_executable(path: &Path) -> Result<bool, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        return fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .map_err(|error| format!("could not inspect {}: {error}", path.display()));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(false)
    }
}

fn unpack_distribution_package(archive: &Path, destination: &Path) -> Result<(), String> {
    fn take<'a>(
        bytes: &'a [u8],
        cursor: &mut usize,
        count: usize,
        archive: &Path,
    ) -> Result<&'a [u8], String> {
        let end = cursor
            .checked_add(count)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("truncated package distribution `{}`", archive.display()))?;
        let value = &bytes[*cursor..end];
        *cursor = end;
        Ok(value)
    }

    let bytes = fs::read(archive)
        .map_err(|error| format!("could not read {}: {error}", archive.display()))?;
    if !bytes.starts_with(b"SEVPKG\0\x02") {
        return Err(format!(
            "unsupported package distribution format in `{}`",
            archive.display()
        ));
    }
    let mut cursor = 8;
    let count = u32::from_be_bytes(
        take(&bytes, &mut cursor, 4, archive)?
            .try_into()
            .expect("four bytes"),
    );
    if count > 100_000 {
        return Err(format!(
            "package distribution `{}` declares too many entries",
            archive.display()
        ));
    }
    let mut paths = BTreeSet::new();
    for _ in 0..count {
        let path_length = u32::from_be_bytes(
            take(&bytes, &mut cursor, 4, archive)?
                .try_into()
                .expect("four bytes"),
        ) as usize;
        let relative = std::str::from_utf8(take(&bytes, &mut cursor, path_length, archive)?)
            .map_err(|_| format!("package `{}` contains a non-UTF-8 path", archive.display()))?;
        let flags = take(&bytes, &mut cursor, 1, archive)?[0];
        if flags & !1 != 0 {
            return Err(format!(
                "package entry `{relative}` has unknown mandatory flags {flags:#x}"
            ));
        }
        let data_length = u64::from_be_bytes(
            take(&bytes, &mut cursor, 8, archive)?
                .try_into()
                .expect("eight bytes"),
        );
        let data_length = usize::try_from(data_length)
            .map_err(|_| format!("package entry `{relative}` is too large for this host"))?;
        let contents = take(&bytes, &mut cursor, data_length, archive)?;
        let path = Path::new(relative);
        let safe = !relative.is_empty()
            && !relative.contains('\\')
            && path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)));
        if !safe || !paths.insert(relative.to_owned()) {
            return Err(format!("unsafe or duplicate package entry `{relative}`"));
        }
        let output = destination.join(path);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        fs::write(&output, contents)
            .map_err(|error| format!("could not write {}: {error}", output.display()))?;
        #[cfg(unix)]
        if flags & 1 != 0 {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&output, fs::Permissions::from_mode(0o755)).map_err(|error| {
                format!("could not mark {} executable: {error}", output.display())
            })?;
        }
    }
    if cursor != bytes.len() {
        return Err(format!(
            "package distribution `{}` has trailing unindexed data",
            archive.display()
        ));
    }
    Ok(())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyEdit {
    Add,
    Remove,
    Update,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RegistryVersion {
    major: u64,
    minor: u64,
    patch: u64,
    spelling: String,
}

fn edit_dependency(
    action: DependencyEdit,
    arguments: Vec<String>,
    catalog: &Catalog,
) -> Result<(), String> {
    if arguments.len() != 1 {
        let command = match action {
            DependencyEdit::Add => "add",
            DependencyEdit::Remove => "remove",
            DependencyEdit::Update => "update",
        };
        return Err(format!("usage: sev {command} <package>[@version]"));
    }
    let (argument_name, requested) = package_specification(&arguments[0])?;
    if action != DependencyEdit::Add && requested.is_some() {
        return Err(
            "version selectors are accepted by `sev add`; use `sev update NAME` afterward".into(),
        );
    }
    let manifest_path = project_manifest_path(Path::new("."))?;
    let original = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("could not read {}: {error}", manifest_path.display()))?;
    let parsed = original
        .parse::<toml::Value>()
        .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
    let (alias, package, registry) = match action {
        DependencyEdit::Add => (argument_name.clone(), argument_name, None),
        DependencyEdit::Remove | DependencyEdit::Update => {
            let (package, registry) = declared_dependency(&parsed, &argument_name)?;
            (argument_name, package, registry)
        }
    };
    let selected = if action == DependencyEdit::Remove {
        None
    } else {
        let requirement = if action == DependencyEdit::Update {
            None
        } else {
            requested.as_deref()
        };
        Some(select_registry_version(
            &package,
            requirement,
            registry.as_deref(),
        )?)
    };

    let mut document = original
        .parse::<toml_edit::DocumentMut>()
        .map_err(|error| format!("invalid {}: {error}", manifest_path.display()))?;
    if document.get("dependencies").is_none() {
        document["dependencies"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let dependencies = document["dependencies"]
        .as_table_mut()
        .ok_or_else(|| "`[dependencies]` must be a table".to_owned())?;
    match action {
        DependencyEdit::Remove => {
            if dependencies.remove(&alias).is_none() {
                return Err(format!("dependency `{alias}` is not declared"));
            }
        }
        DependencyEdit::Add | DependencyEdit::Update => {
            let version = selected.as_ref().expect("non-remove selects a version");
            dependencies.insert(
                &alias,
                dependency_manifest_item(&alias, &package, &version.spelling, registry.as_deref()),
            );
        }
    }

    let lock_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("sev.lock");
    let previous_lock = fs::read(&lock_path).ok();
    write_atomic(&manifest_path, document.to_string().as_bytes())?;
    let result = (|| {
        let manifest = Manifest::load(&manifest_path, catalog)?;
        if action != DependencyEdit::Remove {
            validate_package_targets(&manifest, catalog)?;
        }
        write_atomic(&lock_path, manifest.render_lockfile()?.as_bytes())
    })();
    if let Err(error) = result {
        let _ = write_atomic(&manifest_path, original.as_bytes());
        match previous_lock {
            Some(lock) => {
                let _ = write_atomic(&lock_path, &lock);
            }
            None => {
                let _ = fs::remove_file(&lock_path);
            }
        }
        return Err(error);
    }
    match action {
        DependencyEdit::Add => println!(
            "added {alias} {}",
            selected.expect("add selected a version").spelling
        ),
        DependencyEdit::Update => println!(
            "updated {alias} {}",
            selected.expect("update selected a version").spelling
        ),
        DependencyEdit::Remove => println!("removed {alias}"),
    }
    Ok(())
}

fn install_package(arguments: Vec<String>, catalog: &Catalog) -> Result<(), String> {
    if arguments.len() != 1 {
        return Err("usage: sev install <package>[@version]".into());
    }
    let options = CommonOptions::default();
    let (executable, binary_name) = materialize_registry_binary(&arguments[0], &options, catalog)?;
    let bin_home = std::env::var_os("SEVERIAN_BIN_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_BIN_HOME").map(PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local/bin"))
        })
        .ok_or_else(|| {
            "could not locate the installation bin directory; set SEVERIAN_BIN_HOME".to_owned()
        })?;
    fs::create_dir_all(&bin_home)
        .map_err(|error| format!("could not create {}: {error}", bin_home.display()))?;
    let destination = bin_home.join(&binary_name);
    let temporary = bin_home.join(format!(".{binary_name}.{}.tmp", process::id()));
    fs::copy(&executable, &temporary).map_err(|error| {
        format!(
            "could not copy {} to {}: {error}",
            executable.display(),
            temporary.display()
        )
    })?;
    fs::rename(&temporary, &destination).map_err(|error| {
        format!(
            "could not install {} as {}: {error}",
            executable.display(),
            destination.display()
        )
    })?;
    println!("installed {binary_name} to {}", destination.display());
    Ok(())
}

fn package_specification(value: &str) -> Result<(String, Option<String>), String> {
    let (name, requirement) = value
        .rsplit_once('@')
        .map_or((value, None), |(name, version)| (name, Some(version)));
    if name.is_empty() || name.contains('/') || name.contains('\\') || matches!(name, "." | "..") {
        return Err(format!("invalid package name `{name}`"));
    }
    if requirement.is_some_and(str::is_empty) {
        return Err(format!(
            "package specification `{value}` is missing a version"
        ));
    }
    Ok((name.to_owned(), requirement.map(str::to_owned)))
}

fn declared_dependency(
    manifest: &toml::Value,
    alias: &str,
) -> Result<(String, Option<String>), String> {
    let declaration = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .and_then(|dependencies| dependencies.get(alias))
        .ok_or_else(|| format!("dependency `{alias}` is not declared"))?;
    if declaration.is_str() {
        return Ok((alias.to_owned(), None));
    }
    let detail = declaration
        .as_table()
        .ok_or_else(|| format!("dependency `{alias}` has an invalid declaration"))?;
    let package = detail
        .get("package")
        .and_then(toml::Value::as_str)
        .unwrap_or(alias)
        .to_owned();
    let registry = detail
        .get("registry")
        .and_then(toml::Value::as_str)
        .map(str::to_owned);
    if detail.get("path").is_some() {
        return Err(format!(
            "dependency `{alias}` is a path override; replace it with a registry dependency before updating"
        ));
    }
    Ok((package, registry))
}

fn select_registry_version(
    package: &str,
    requirement: Option<&str>,
    registry: Option<&str>,
) -> Result<RegistryVersion, String> {
    let root = registry_root(registry)?;
    let versions = root.join("packages").join(package);
    let entries = fs::read_dir(&versions).map_err(|error| {
        format!(
            "package `{package}` was not found in registry `{}`: {error}",
            root.display()
        )
    })?;
    let prefix = requirement
        .map(parse_version_prefix)
        .transpose()?
        .unwrap_or_default();
    let mut candidates = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let release = entry.path();
            let version = entry.file_name().to_string_lossy().into_owned();
            release.join("metadata/package.toml").is_file()
                || release.join("source/package.toml").is_file()
                || release.join(format!("{package}-{version}.pkg")).is_file()
        })
        .filter_map(|entry| {
            let spelling = entry.file_name().to_string_lossy().into_owned();
            RegistryVersion::parse(&spelling).ok()
        })
        .filter(|version| version.matches_prefix(&prefix))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        let requested = requirement
            .map(|value| format!(" matching `{value}`"))
            .unwrap_or_default();
        format!(
            "package `{package}` has no published version{requested} in registry `{}`",
            root.display()
        )
    })
}

impl RegistryVersion {
    fn parse(value: &str) -> Result<Self, String> {
        let parts = parse_version_prefix(value)?;
        if parts.len() != 3 {
            return Err(format!(
                "published version `{value}` is not an exact major.minor.patch version"
            ));
        }
        Ok(Self {
            major: parts[0],
            minor: parts[1],
            patch: parts[2],
            spelling: value.to_owned(),
        })
    }

    fn matches_prefix(&self, prefix: &[u64]) -> bool {
        [self.major, self.minor, self.patch]
            .iter()
            .zip(prefix)
            .all(|(actual, expected)| actual == expected)
    }
}

fn parse_version_prefix(value: &str) -> Result<Vec<u64>, String> {
    let parts = value
        .split('.')
        .map(|part| {
            part.parse::<u64>()
                .map_err(|_| format!("invalid version selector `{value}`"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("invalid version selector `{value}`"));
    }
    Ok(parts)
}

fn dependency_manifest_item(
    alias: &str,
    package: &str,
    version: &str,
    registry: Option<&str>,
) -> toml_edit::Item {
    if alias == package && registry.is_none_or(|registry| registry == "default") {
        return toml_edit::value(version);
    }
    let mut detail = toml_edit::InlineTable::new();
    detail.insert("package", toml_edit::Value::from(package));
    detail.insert("version", toml_edit::Value::from(version));
    if let Some(registry) = registry {
        detail.insert("registry", toml_edit::Value::from(registry));
    }
    toml_edit::Item::Value(toml_edit::Value::InlineTable(detail))
}

fn project_manifest_path(start: &Path) -> Result<PathBuf, String> {
    let mut directory = fs::canonicalize(start)
        .map_err(|error| format!("could not resolve {}: {error}", start.display()))?;
    loop {
        let manifest = directory.join("package.toml");
        if manifest.is_file() {
            return Ok(manifest);
        }
        directory = directory.parent().map(Path::to_owned).ok_or_else(|| {
            format!(
                "no package.toml found from `{}` to the filesystem root",
                start.display()
            )
        })?;
    }
}

fn validate_package_targets(manifest: &Manifest, catalog: &Catalog) -> Result<(), String> {
    let config = resolve_config(catalog, Some(manifest), &CommonOptions::default())?;
    let compiler = compiler(&config, Some(manifest), false)?;
    if let Some(library) = &manifest.library {
        compiler
            .check_file(&library.path)
            .map_err(|error| error.to_string())?;
    }
    for binary in &manifest.bins {
        compiler
            .check_file(&binary.path)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", process::id()));
    fs::write(&temporary, contents)
        .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("could not replace {}: {error}", path.display()))
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
commands:\n  check   build   compile   run   test   doctor   api <list|show|check|diff>   publish   add   remove   update   install   new   init   config <show|sync|defaults>\n\n\
package lifecycle:\n  sev add NAME[@VERSION]       Add a project dependency and refresh sev.lock.\n  sev remove NAME              Remove a project dependency and refresh sev.lock.\n  sev update NAME              Resolve the newest package and refresh sev.lock.\n  sev run NAME[@VERSION]       Resolve temporarily and run now.\n  sev install NAME[@VERSION]   Install a package executable for the machine.\n\n\
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
        "  --bin NAME  Select a package binary.\n  --emit STAGE  Print ast, hir, mir, lir, or mlir, or write agent-ir; do not execute.\n  -o PATH     Write the selected artifact, emitted IR, or Agent IR directory to PATH.\n\ntest options:\n  --mutate    Run mutation testing.\n",
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
    fn doctor_reports_optional_backends_without_provisioning_them() {
        doctor(Vec::new()).unwrap();
        assert_eq!(
            doctor(vec!["unexpected".into()]).unwrap_err(),
            "usage: sev doctor"
        );
    }

    #[test]
    fn emit_accepts_separate_and_equals_forms() {
        let separate =
            parse_common(vec!["hello.sev".into(), "--emit".into(), "lir".into()]).unwrap();
        assert_eq!(separate.emit, Some(EmitStage::Lir));

        let equals = parse_common(vec!["hello.sev".into(), "--emit=mlir".into()]).unwrap();
        assert_eq!(equals.emit, Some(EmitStage::Mlir));

        let agent = parse_common(vec!["hello.sev".into(), "--emit=agent-ir".into()]).unwrap();
        assert_eq!(agent.emit, Some(EmitStage::AgentIr));
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

    #[test]
    fn distribution_reader_rejects_parent_path_entries() {
        let root = env::temp_dir().join(format!("severian-unsafe-package-{}", process::id()));
        let archive = root.join("unsafe.pkg");
        let destination = root.join("output");
        fs::create_dir_all(&root).unwrap();
        let relative = b"../escaped";
        let mut bytes = b"SEVPKG\0\x02".to_vec();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&(relative.len() as u32).to_be_bytes());
        bytes.extend_from_slice(relative);
        bytes.push(0);
        bytes.extend_from_slice(&0u64.to_be_bytes());
        fs::write(&archive, bytes).unwrap();
        fs::create_dir_all(&destination).unwrap();

        let error = unpack_distribution_package(&archive, &destination).unwrap_err();
        assert!(error.contains("unsafe or duplicate package entry"));
        assert!(!root.join("escaped").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
