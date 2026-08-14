mod architecture_command;
mod build_options;
mod coverage_policy;
mod error_catalog;
mod runtime_diagnostics;
mod scaffold;

use severian_backend::{NativeCompileOptions, NativeSanitizer};
use severian_driver::build_cache::BuildGateCache;
use severian_driver::{
    check_path, compile_dependency_path, compile_native, compile_native_integration_tests,
    compile_native_profile_tests, compile_native_tests, compile_native_with_options, compile_path,
    inspect_toolchain, native_coverage_test_compilation, native_integration_test_count,
    native_profile_test_compilation, native_profile_test_count, native_test_compilation,
    native_test_count, Compilation,
};
use severian_package::{BinaryTarget, BuildGate, BuildPolicy};
use std::{
    collections::HashSet,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn main() {
    if let Err(error) = execute(std::env::args().skip(1).collect()) {
        if error.starts_with("error[E") {
            eprintln!("{error}");
        } else {
            eprintln!("error: {error}");
        }
        std::process::exit(1);
    }
}

fn execute(args: Vec<String>) -> Result<(), String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage());
    };

    if command.ends_with(".sev") {
        if args.len() != 1 {
            return Err(usage());
        }
        return run_targets(Path::new(command), &[], None);
    }

    match command {
        "--emit" if args.len() == 3 => {
            let mut build_args = vec![args[2].clone(), "--emit".into(), args[1].clone()];
            if args[1] == "stablehlo" {
                build_args.extend(["--target".into(), "xla".into()]);
            }
            build_command(&build_args).map(|_| ())
        }
        "doctor" if args.len() == 1 => doctor(),
        "new" if args.len() == 2 => new_project(Path::new(&args[1])),
        "init" if args.len() <= 2 => init_project(args.get(1).map_or(Path::new("."), Path::new)),
        "add" => add_command(&args[1..]),
        "remove" if args.len() == 2 => remove_command(&args[1]),
        "update" if args.len() <= 2 => {
            update_command(args.get(1).map_or(Path::new("."), Path::new))
        }
        "publish" if args.len() <= 2 => {
            publish_command(args.get(1).map_or(Path::new("."), Path::new))
        }
        "install" => install_command(&args[1..]),
        "trust" => trust_command(&args[1..]),
        "verify" if args.len() <= 2 => {
            verify_command(args.get(1).map_or(Path::new("."), Path::new))
        }
        "check" if args.len() <= 2 => check_targets(args.get(1).map_or(Path::new("."), Path::new)),
        "architecture" => architecture_command::run(&args[1..]),
        "lint" => lint_command(&args[1..]),
        "fmt" => fmt_command(&args[1..]),
        "build" => build_command(&args[1..]).map(|_| ()),
        "run" => run_command(&args[1..]),
        "test" => test_command(&args[1..]),
        "debug" if args.len() <= 2 => debug_targets(args.get(1).map_or(Path::new("."), Path::new)),
        "coverage" if args.len() == 2 => coverage(Path::new(&args[1])),
        "memory" => memory_command(&args[1..]),
        "kernel" => kernel_command(&args[1..]),
        "clean" if args.len() <= 2 => clean(args.get(1).map(Path::new)),
        "tree" if args.len() == 2 => tree(Path::new(&args[1])),
        "metadata" if args.len() == 2 => metadata(Path::new(&args[1])),
        "explain" if args.len() == 2 => explain(&args[1]),
        "errors" if args.len() == 1 => error_catalog::print(),
        "emit-mlir" if args.len() == 2 => emit_stdout(Path::new(&args[1]), EmitMode::Mlir),
        "compile" if args.len() == 2 || args.len() == 4 => legacy_compile(&args),
        "compile-tests" if args.len() == 4 && args[2] == "-o" => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            let count = compile_native_tests(&compilation, Path::new(&args[3]))
                .map_err(|error| error.to_string())?;
            println!("{} ({count} native tests)", args[3]);
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        _ => Err(usage()),
    }
}

fn kernel_command(args: &[String]) -> Result<(), String> {
    use severian_lowering::kernel::{
        emit_stablehlo, emit_triton_ir, find, select_backend, KernelBackend, KernelTarget,
    };

    let action = args.first().map(String::as_str).ok_or_else(kernel_usage)?;
    if action != "inspect" && action != "emit" {
        return Err(kernel_usage());
    }
    let source = args
        .get(1)
        .filter(|value| !value.starts_with('-'))
        .map(PathBuf::from)
        .ok_or_else(kernel_usage)?;
    let mut entry = None;
    let mut backend = None;
    let mut target = KernelTarget::Gpu;
    let mut output = None;
    let mut index = 2;
    while index < args.len() {
        let option = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("option `{option}` requires a value"))?;
        match option {
            "--entry" => entry = Some(value.clone()),
            "--backend" => {
                backend = Some(match value.as_str() {
                    "auto" => KernelBackend::Auto,
                    "xla" => KernelBackend::Xla,
                    "triton" => KernelBackend::Triton,
                    "llvm" => KernelBackend::Llvm,
                    _ => return Err(format!("unknown kernel backend `{value}`")),
                })
            }
            "--target" => {
                target = KernelTarget::parse(value)
                    .map_err(|error| format!("invalid kernel target `{value}`: {error}"))?;
            }
            "--output" if action == "emit" => output = Some(PathBuf::from(value)),
            _ => {
                return Err(format!(
                    "unknown kernel option `{option}`\n{}",
                    kernel_usage()
                ))
            }
        }
        index += 2;
    }

    let compilation = compile_path(&source).map_err(|error| error.to_string())?;
    let kernel = find(&compilation.mir, entry.as_deref()).map_err(|error| error.to_string())?;
    let selection = select_backend(&kernel, backend.unwrap_or(kernel.policy), target)
        .map_err(|error| error.to_string())?;
    if action == "inspect" {
        println!("kernel: {}", kernel.name);
        println!("operation: {}", kernel.operation.name());
        println!("target: {}", selection.target.name());
        println!("requested backend: {}", selection.requested.name());
        println!("selected backend: {}", selection.selected.name());
        println!(
            "fallback: {}",
            selection.fallback.map_or("none", KernelBackend::name)
        );
        println!("reason: {}", selection.reason);
        println!("parameters: {:?}", kernel.parameters);
        println!("result: {:?}", kernel.result);
        return Ok(());
    }

    let artifact = match selection.selected {
        KernelBackend::Triton => emit_triton_ir(&kernel).map_err(|error| error.to_string())?,
        KernelBackend::Xla => emit_stablehlo(&kernel)
            .map_err(|error| error.to_string())?
            .as_str()
            .to_string(),
        KernelBackend::Llvm => {
            return Err(
                "per-kernel LLVM emission is not available yet; use `sev build --emit llvm --target native` for the whole native module"
                    .into(),
            )
        }
        KernelBackend::Auto => unreachable!("backend selection must resolve auto"),
    };
    if let Some(output) = output {
        if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(&output, artifact).map_err(|error| error.to_string())?;
        println!(
            "Emitted {} kernel {} -> {}",
            selection.selected.name(),
            kernel.name,
            output.display()
        );
    } else {
        print!("{artifact}");
    }
    Ok(())
}

fn kernel_usage() -> String {
    "usage:\n  sev kernel inspect <source.sev> [--entry NAME] [--backend auto|xla|triton|llvm] [--target cpu|gpu|tpu|nvidia|amd|cuda:sm_NN|rocm:gfxNNNN]\n  sev kernel emit <source.sev> [--entry NAME] [--backend auto|xla|triton] [--target gpu|tpu|nvidia|amd|cuda:sm_NN|rocm:gfxNNNN] [--output PATH]".into()
}

fn add_command(args: &[String]) -> Result<(), String> {
    let Some(name) = args.first().filter(|value| !value.starts_with('-')) else {
        return Err("usage: sev add <package> [--version REQUIREMENT] [--path PATH]".into());
    };
    let mut version = None;
    let mut path = None;
    let mut published_name = None;
    let mut index = 1;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("option `{}` requires a value", args[index]))?;
        match args[index].as_str() {
            "--version" => version = Some(value.clone()),
            "--path" => path = Some(value.clone()),
            "--package" => published_name = Some(value.clone()),
            option => return Err(format!("unknown add option `{option}`")),
        }
        index += 2;
    }
    let manifest_path = project_manifest(Path::new("."))?;
    let original = fs::read_to_string(&manifest_path).map_err(|error| error.to_string())?;
    let mut manifest = read_manifest_value(&manifest_path)?;
    let table = manifest
        .as_table_mut()
        .ok_or_else(|| format!("{} is not a TOML table", manifest_path.display()))?;
    let dependencies = table
        .entry("dependencies")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .ok_or("[dependencies] must be a table")?;
    if dependencies.contains_key(name) {
        return Err(format!("dependency `{name}` already exists"));
    }
    let specification = if let Some(path) = path {
        let mut detail = toml::Table::new();
        detail.insert("path".into(), toml::Value::String(path));
        if let Some(version) = version {
            detail.insert("version".into(), toml::Value::String(version));
        }
        if let Some(package) = published_name {
            detail.insert("package".into(), toml::Value::String(package));
        }
        toml::Value::Table(detail)
    } else if published_name.is_some() {
        let mut detail = toml::Table::new();
        detail.insert(
            "version".into(),
            toml::Value::String(version.unwrap_or_else(|| "*".into())),
        );
        detail.insert(
            "package".into(),
            toml::Value::String(published_name.expect("published name is present")),
        );
        toml::Value::Table(detail)
    } else {
        toml::Value::String(version.unwrap_or_else(|| "*".into()))
    };
    dependencies.insert(name.clone(), specification);
    write_manifest_value(&manifest_path, &manifest)?;
    if let Err(error) = severian_package::resolve_dependencies(&manifest_path) {
        fs::write(&manifest_path, original).map_err(|rollback| {
            format!(
                "dependency resolution failed ({error}) and restoring {} also failed: {rollback}",
                manifest_path.display()
            )
        })?;
        return Err(error.to_string());
    }
    println!("Added {name}");
    Ok(())
}

fn remove_command(name: &str) -> Result<(), String> {
    let manifest_path = project_manifest(Path::new("."))?;
    let mut manifest = read_manifest_value(&manifest_path)?;
    let removed = manifest
        .get_mut("dependencies")
        .and_then(toml::Value::as_table_mut)
        .and_then(|dependencies| dependencies.remove(name));
    if removed.is_none() {
        return Err(format!("dependency `{name}` is not declared"));
    }
    write_manifest_value(&manifest_path, &manifest)?;
    // Regenerate the lockfile immediately so removed packages do not remain
    // authoritative merely because the next command is not a build.
    severian_package::resolve_dependencies(&manifest_path).map_err(|error| error.to_string())?;
    println!("Removed {name}");
    Ok(())
}

fn update_command(input: &Path) -> Result<(), String> {
    let manifest = project_manifest(input)?;
    let resolution =
        severian_package::update_dependencies(&manifest).map_err(|error| error.to_string())?;
    for dependency in &resolution.dependencies {
        println!(
            "Resolved {} -> {} {}",
            dependency.import_name, dependency.package_name, dependency.version
        );
    }
    println!("Updated {}", resolution.lockfile.display());
    Ok(())
}

fn publish_command(input: &Path) -> Result<(), String> {
    let manifest = project_manifest(input)?;
    severian_package::resolve_dependencies(&manifest).map_err(|error| error.to_string())?;
    let package =
        severian_package::publish_package(&manifest, None).map_err(|error| error.to_string())?;
    println!(
        "Published {} {} ({})",
        package.package_name,
        package.version,
        package.checksum.as_deref().unwrap_or("unverified")
    );
    Ok(())
}

fn install_command(args: &[String]) -> Result<(), String> {
    if args.first().is_none_or(|value| value.starts_with('-')) {
        return install_project_command(args);
    }
    install_binary_command(args)
}

fn install_project_command(args: &[String]) -> Result<(), String> {
    let mut dry_run = false;
    let mut locked = false;
    for option in args {
        match option.as_str() {
            "--dry-run" => dry_run = true,
            "--locked" => locked = true,
            _ => return Err(format!("unknown project install option `{option}`")),
        }
    }
    let manifest = project_manifest(Path::new("."))?;
    let plan = severian_package::plan_installation(&manifest, locked)
        .map_err(|error| error.to_string())?;
    if plan.items.is_empty() {
        println!(
            "Package {} has no external installation requirements.",
            plan.package_name
        );
        if !dry_run {
            severian_package::perform_installation(&plan).map_err(|error| error.to_string())?;
            println!("Updated {}", plan.lockfile.display());
        }
        return Ok(());
    }
    println!("Package {} requires:\n", plan.package_name);
    for item in &plan.items {
        println!("  {} {}", item.locked.name, item.locked.version);
        println!("  Publisher: {}", item.locked.publisher);
        let domain = item
            .locked
            .source
            .strip_prefix("https://")
            .and_then(|source| source.split('/').next())
            .unwrap_or(&item.locked.source);
        println!("  Source: {domain}");
        println!("  Trust valid through: {}", item.locked.trusted_until);
        println!(
            "  System installation: {}\n",
            if item.system_install { "yes" } else { "no" }
        );
    }
    if dry_run {
        println!("Dry run: no artifacts installed and sev.lock was not changed by the installer.");
        return Ok(());
    }
    print!("Install? [y/N] ");
    io::stdout().flush().map_err(|error| error.to_string())?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .map_err(|error| error.to_string())?;
    if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES" | "Yes") {
        println!("Installation cancelled; no external artifacts were changed.");
        return Ok(());
    }
    severian_package::perform_installation(&plan).map_err(|error| error.to_string())?;
    println!(
        "Installed verified external requirements and updated {}",
        plan.lockfile.display()
    );
    Ok(())
}

fn install_binary_command(args: &[String]) -> Result<(), String> {
    let Some(name) = args.first().filter(|name| !name.starts_with('-')) else {
        return Err("usage: sev install <package> [--version REQUIREMENT]".into());
    };
    let mut version = "*".to_owned();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--version" if index + 1 < args.len() => {
                version = args[index + 1].clone();
                index += 2;
            }
            option => return Err(format!("unknown install option `{option}`")),
        }
    }
    let temporary = std::env::temp_dir().join(format!(
        "severian-install-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos()
    ));
    fs::create_dir_all(&temporary).map_err(|error| error.to_string())?;
    let result = (|| {
        let manifest = temporary.join(severian_package::MANIFEST_FILE);
        fs::write(
            &manifest,
            format!(
                "[package]\nname = \"severian-install\"\nversion = \"0.0.0\"\n\n[dependencies]\ninstalled_package = {{ package = {:?}, version = {:?} }}\n",
                name, version
            ),
        )
        .map_err(|error| error.to_string())?;
        let resolution =
            severian_package::resolve_dependencies(&manifest).map_err(|error| error.to_string())?;
        let package = resolution
            .dependencies
            .iter()
            .find(|dependency| dependency.import_name == "installed_package")
            .ok_or_else(|| format!("package `{name}` did not resolve"))?;
        let target = severian_package::default_binary_target(&package.root)
            .map_err(|error| error.to_string())?;
        let mut libraries = HashSet::new();
        build_libraries(&target.source, &mut libraries)?;
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        let home = std::env::var_os("SEVERIAN_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".sev")))
            .unwrap_or_else(|| PathBuf::from(".sev"));
        let binary_directory = home.join("bin");
        fs::create_dir_all(&binary_directory).map_err(|error| error.to_string())?;
        let output = binary_directory.join(&target.name);
        compile_native(&compilation, &output).map_err(|error| error.to_string())?;
        println!(
            "Installed {} {} -> {}",
            name,
            package.version,
            output.display()
        );
        Ok(())
    })();
    let _ = fs::remove_dir_all(&temporary);
    result
}

fn trust_command(args: &[String]) -> Result<(), String> {
    let registry =
        severian_package::TrustRegistry::load_default().map_err(|error| error.to_string())?;
    match args {
        [command] if command == "list" => {
            if registry.publishers.is_empty() {
                println!("No trusted publishers are configured.");
                return Ok(());
            }
            for publisher in registry.publishers {
                println!(
                    "{}  {}..{}  system-install={}",
                    publisher.name,
                    publisher.trusted_from.as_str(),
                    publisher.trusted_until.as_str(),
                    publisher.allow_system_install
                );
            }
            Ok(())
        }
        [command, name] if command == "show" => {
            let publisher = registry
                .publisher(name)
                .map_err(|error| error.to_string())?;
            println!("Publisher: {}", publisher.name);
            println!("Allowed domains: {}", publisher.allowed_domains.join(", "));
            println!(
                "Package namespaces: {}",
                publisher.package_namespaces.join(", ")
            );
            println!("Signing keys: {}", publisher.signing_keys.len());
            println!("Trusted from: {}", publisher.trusted_from.as_str());
            println!("Trusted until: {}", publisher.trusted_until.as_str());
            println!("System installation: {}", publisher.allow_system_install);
            Ok(())
        }
        _ => Err("usage:\n  sev trust list\n  sev trust show <publisher>".into()),
    }
}

fn verify_command(input: &Path) -> Result<(), String> {
    let manifest = project_manifest(input)?;
    let plan =
        severian_package::verify_installation(&manifest).map_err(|error| error.to_string())?;
    println!(
        "Verified {} package dependencies and {} external artifacts against {}",
        severian_package::resolve_dependencies_transient(&manifest)
            .map_err(|error| error.to_string())?
            .dependencies
            .len(),
        plan.items.len(),
        plan.lockfile.display()
    );
    Ok(())
}

fn project_manifest(input: &Path) -> Result<PathBuf, String> {
    if input.is_file() {
        if input.file_name().and_then(|name| name.to_str()) == Some(severian_package::MANIFEST_FILE)
        {
            return Ok(input.to_path_buf());
        }
        return severian_package::find_manifest(input)
            .ok_or_else(|| format!("could not find package.toml from {}", input.display()));
    }
    severian_package::nearest_manifest(input)
        .ok_or_else(|| format!("could not find package.toml from {}", input.display()))
}

fn read_manifest_value(path: &Path) -> Result<toml::Value, String> {
    let source = fs::read_to_string(path).map_err(|error| error.to_string())?;
    toml::from_str(&source).map_err(|error| format!("invalid manifest {}: {error}", path.display()))
}

fn write_manifest_value(path: &Path, manifest: &toml::Value) -> Result<(), String> {
    let source = toml::to_string_pretty(manifest).map_err(|error| error.to_string())?;
    fs::write(path, source).map_err(|error| error.to_string())
}

fn new_project(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "cannot create `{}` because it already exists",
            path.display()
        ));
    }
    fs::create_dir_all(path).map_err(|error| error.to_string())?;
    init_project(path)
}

fn init_project(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| error.to_string())?;
    if path.join("package.toml").exists() {
        return Err(format!("{} already contains package.toml", path.display()));
    }
    let canonical = path.canonicalize().map_err(|error| error.to_string())?;
    let name = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("severian-app")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    fs::write(path.join("package.toml"), scaffold::package_manifest(&name))
        .map_err(|error| error.to_string())?;
    fs::write(path.join("sev.lock"), "version = 1\npackages = []\n")
        .map_err(|error| error.to_string())?;
    let source_directory = path.join("src");
    fs::create_dir_all(&source_directory).map_err(|error| error.to_string())?;
    let main = source_directory.join("main.sev");
    if !main.exists() {
        fs::write(
            &main,
            "def main():\n    print(\"hello, severian\")\n\ntest \"entrypoint\":\n    main()\n",
        )
        .map_err(|error| error.to_string())?;
    }
    println!("Initialized {}", path.display());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmitMode {
    Executable,
    Hir,
    Mir,
    Mlir,
    StableHlo,
    Llvm,
    Asm,
}

impl EmitMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "executable" | "binary" => Ok(Self::Executable),
            "hir" => Ok(Self::Hir),
            "mir" => Ok(Self::Mir),
            "mlir" => Ok(Self::Mlir),
            "stablehlo" => Ok(Self::StableHlo),
            "llvm" | "llvm-ir" => Ok(Self::Llvm),
            "asm" | "assembly" => Ok(Self::Asm),
            _ => Err(format!("unknown emit format `{value}`")),
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Executable => {
                if cfg!(windows) {
                    "exe"
                } else {
                    ""
                }
            }
            Self::Hir => "hir",
            Self::Mir => "mir",
            Self::Mlir | Self::StableHlo => "mlir",
            Self::Llvm => "ll",
            Self::Asm => "s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildTarget {
    Native,
    Xla,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageFormat {
    Text,
    Json,
}

fn parse_build_target(value: &str) -> Result<BuildTarget, String> {
    match value {
        "native" | "cpu" => Ok(BuildTarget::Native),
        value if value == "xla" || value.starts_with("xla:") => Ok(BuildTarget::Xla),
        _ => Err(format!(
            "unsupported build target `{value}`; use native or xla"
        )),
    }
}

fn parse_message_format(value: &str) -> Result<MessageFormat, String> {
    match value {
        "text" => Ok(MessageFormat::Text),
        "json" => Ok(MessageFormat::Json),
        _ => Err(format!(
            "unknown message format `{value}`; use text or json"
        )),
    }
}

#[derive(Debug)]
struct BuildFinding {
    severity: &'static str,
    code: String,
    path: PathBuf,
    rendered: String,
    diagnostic: Option<severian_diagnostics::Diagnostic>,
}

fn build_command(args: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut input = PathBuf::from(".");
    let mut index = 0;
    if args.first().is_some_and(|value| !value.starts_with('-')) {
        input = PathBuf::from(&args[0]);
        index = 1;
    }
    let configured = build_options::load(&input)?;
    let mut emit = EmitMode::parse(&configured.emit)?;
    let mut target = parse_build_target(&configured.target)?;
    let mut max_errors = configured.max_errors;
    let mut message_format = parse_message_format(&configured.message_format)?;
    let mut verify_each = configured.verify_each;
    let mut diagnostics = configured.diagnostics;
    while index < args.len() {
        match args[index].as_str() {
            "--emit" if index + 1 < args.len() => {
                emit = EmitMode::parse(&args[index + 1])?;
                index += 2;
            }
            "--target" if index + 1 < args.len() => {
                target = parse_build_target(&args[index + 1])?;
                index += 2;
            }
            "--max-errors" if index + 1 < args.len() => {
                max_errors = args[index + 1]
                    .parse::<usize>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "--max-errors must be a positive integer".to_string())?;
                index += 2;
            }
            "--message-format" if index + 1 < args.len() => {
                message_format = parse_message_format(&args[index + 1])?;
                index += 2;
            }
            "--diagnostics" if index + 1 < args.len() => {
                diagnostics = build_options::DiagnosticsMode::parse(&args[index + 1])?;
                index += 2;
            }
            value if value.starts_with("--diagnostics=") => {
                diagnostics = build_options::DiagnosticsMode::parse(
                    value.trim_start_matches("--diagnostics="),
                )?;
                index += 1;
            }
            "--verify-each" => {
                verify_each = true;
                index += 1;
            }
            value => return Err(format!("unknown build option `{value}`\n{}", usage())),
        }
    }
    if target == BuildTarget::Native && emit == EmitMode::StableHlo {
        return Err("StableHLO emission requires `--target xla`".into());
    }

    let policy = BuildPolicy::for_input(&input).map_err(|error| error.to_string())?;
    let targets = resolve_targets(&input)?;
    let gate_cache = BuildGateCache::discover(&policy.root, &input)?;
    build_progress(
        message_format,
        &format!(
            "Build policy: {}",
            policy
                .pipeline
                .iter()
                .map(|gate| gate.name())
                .collect::<Vec<_>>()
                .join(" -> ")
        ),
    );
    if gate_cache.is_fresh(BuildGate::Compile) {
        build_progress(message_format, "[compile] CACHED");
    } else {
        gate_cache.invalidate_from(BuildGate::Compile, &policy.pipeline)?;
        build_progress(message_format, "[compile] RUN");
        let findings = collect_build_findings(&input, max_errors, diagnostics)?;
        render_build_findings(&findings, message_format);
        let error_count = findings
            .iter()
            .filter(|finding| finding.severity == "error")
            .count();
        if error_count > 0 {
            return Err(format!(
                "build stopped after {error_count} independent error(s); no artifacts were emitted"
            ));
        }
        gate_cache.record(BuildGate::Compile)?;
        build_progress(message_format, "[compile] PASS");
    }

    for gate in policy.pipeline.iter().copied().skip(1) {
        if gate_cache.is_fresh(gate) {
            build_progress(message_format, &format!("[{}] CACHED", gate.name()));
            continue;
        }
        gate_cache.invalidate_from(gate, &policy.pipeline)?;
        build_progress(message_format, &format!("[{}] RUN", gate.name()));
        match gate {
            BuildGate::Compile => unreachable!("compile is the first and unique gate"),
            BuildGate::Architecture => architecture_command::enforce(&policy)?,
            BuildGate::Test => test_targets(&input, false)?,
            BuildGate::Profile => test_targets(&input, true)?,
            BuildGate::Coverage => coverage_with_policy(&input, &policy)?,
            BuildGate::Memory => {
                memory_test_targets(&input, false, Vec::new(), policy.memory.leaks)?
            }
            BuildGate::Integration => integration_test_targets(&input)?,
        }
        gate_cache.record(gate)?;
        build_progress(message_format, &format!("[{}] PASS", gate.name()));
    }
    let mut libraries = HashSet::new();
    let mut artifacts = Vec::new();
    for target_spec in targets {
        build_libraries(&target_spec.source, &mut libraries)?;
        let compilation = compile_path(&target_spec.source).map_err(|error| error.to_string())?;
        if verify_each {
            println!(
                "Verified {}: resolved HIR, linked HIR, every HIR transformation, and MIR",
                target_spec.source.display()
            );
        }
        let output = artifact_path(&target_spec, emit);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        if emit == EmitMode::Executable && compilation.hir.main().is_none() {
            for output in emit_non_executable_module(&compilation, &target_spec)? {
                println!("Built {} -> {}", target_spec.name, output.display());
                artifacts.push(output);
            }
        } else {
            emit_artifact(&compilation, emit, target, &output, diagnostics)?;
            println!("Built {} -> {}", target_spec.name, output.display());
            artifacts.push(output);
        }
    }
    println!("BUILD PASS");
    Ok(artifacts)
}

fn build_progress(format: MessageFormat, message: &str) {
    match format {
        MessageFormat::Text => println!("{message}"),
        MessageFormat::Json => eprintln!("{message}"),
    }
}

fn collect_build_findings(
    input: &Path,
    max_errors: usize,
    diagnostics_mode: build_options::DiagnosticsMode,
) -> Result<Vec<BuildFinding>, String> {
    let mut sources = if input.is_file() {
        vec![input.to_path_buf()]
    } else if severian_package::nearest_manifest(input).is_some()
        || input.join("main.sev").is_file()
    {
        resolve_targets(input)?
            .into_iter()
            .map(|target| target.source)
            .collect()
    } else {
        let mut sources = Vec::new();
        collect_sources(input, &mut sources).map_err(|error| error.to_string())?;
        sources.sort();
        sources
    };
    sources.dedup();

    let mut findings = Vec::new();
    let mut errors = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for source_path in sources {
        if errors >= max_errors {
            break;
        }
        match compile_path(&source_path) {
            Ok(compilation) => {
                let source = fs::read_to_string(&source_path).map_err(|error| error.to_string())?;
                let diagnostics = severian_diagnostics::lint::run_with_source(
                    &compilation.hir,
                    &Default::default(),
                    &source_path,
                    &source,
                );
                if !diagnostics.diagnostics().is_empty() {
                    let mut source_map = severian_source::SourceMap::new();
                    source_map.add(source_path.clone(), source);
                    for diagnostic in diagnostics.diagnostics() {
                        let mut rendered = severian_diagnostics::render::render(
                            diagnostic,
                            Some(&source_map),
                            &severian_diagnostics::render::RenderOptions {
                                color: false,
                                ..Default::default()
                            },
                        );
                        if diagnostics_mode.is_internal() {
                            rendered.push_str(&format!("\n internal: {diagnostic:#?}"));
                        }
                        let key = format!("{}:{rendered}", source_path.display());
                        if seen.insert(key) {
                            findings.push(BuildFinding {
                                severity: if diagnostic.severity
                                    >= severian_diagnostics::Severity::Error
                                {
                                    "error"
                                } else {
                                    "warning"
                                },
                                code: diagnostic.code.0.clone(),
                                path: source_path.clone(),
                                rendered,
                                diagnostic: Some(diagnostic.clone()),
                            });
                        }
                    }
                }
            }
            Err(error) => {
                let diagnostic = error.diagnostic();
                let mut rendered = error.to_string();
                if diagnostics_mode.is_internal() {
                    rendered.push_str(&format!("\n internal: {error:#?}"));
                }
                let key = format!("{}:{rendered}", source_path.display());
                if seen.insert(key) {
                    findings.push(BuildFinding {
                        severity: "error",
                        code: diagnostic_code(&rendered).unwrap_or("compiler").to_owned(),
                        path: source_path,
                        rendered,
                        diagnostic,
                    });
                    errors += 1;
                }
            }
        }
    }
    Ok(findings)
}

fn render_build_findings(findings: &[BuildFinding], format: MessageFormat) {
    match format {
        MessageFormat::Text => {
            for finding in findings {
                eprintln!("{}", finding.rendered);
            }
        }
        MessageFormat::Json => {
            let entries = findings
                .iter()
                .map(|finding| {
                    finding.diagnostic.as_ref().map_or_else(
                        || {
                            format!(
                                "{{\"severity\":\"{}\",\"code\":\"{}\",\"path\":\"{}\",\"message\":\"{}\",\"rendered\":\"{}\",\"help\":null,\"labels\":[],\"notes\":[],\"suggestions\":[],\"related\":[]}}",
                                finding.severity,
                                json_escape(&finding.code),
                                json_escape(&finding.path.display().to_string()),
                                json_escape(&finding.rendered),
                                json_escape(&finding.rendered),
                            )
                        },
                        |diagnostic| {
                            severian_diagnostics::render::render_json(
                                diagnostic,
                                &finding.rendered,
                                &finding.path,
                            )
                        },
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            println!("[{}]", entries);
        }
    }
}

fn diagnostic_code(message: &str) -> Option<&str> {
    message
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '-'))
        .find(|part| {
            matches!(
                (part.as_bytes().first(), part.len()),
                (Some(b'E'), 7) | (Some(b'W'), 4)
            ) && part.as_bytes()[1..].iter().all(u8::is_ascii_digit)
        })
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write;
                write!(escaped, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn emit_artifact(
    compilation: &Compilation,
    emit: EmitMode,
    target: BuildTarget,
    output: &Path,
    diagnostics: build_options::DiagnosticsMode,
) -> Result<(), String> {
    match emit {
        EmitMode::Executable => {
            compile_native(compilation, output).map_err(|error| build_error(error, diagnostics))
        }
        EmitMode::Hir => fs::write(output, format!("{:#?}", compilation.optimized_hir))
            .map_err(|error| error.to_string()),
        EmitMode::Mir => {
            fs::write(output, format!("{:#?}", compilation.mir)).map_err(|error| error.to_string())
        }
        EmitMode::Mlir => {
            fs::write(output, compilation.mlir.as_str()).map_err(|error| error.to_string())
        }
        EmitMode::StableHlo => {
            if target != BuildTarget::Xla {
                return Err("StableHLO requires the XLA target".into());
            }
            let module = severian_lowering::stablehlo::lower_program(&compilation.optimized_hir)
                .map_err(|error| build_error(error, diagnostics))?;
            fs::write(output, module.as_str()).map_err(|error| error.to_string())
        }
        EmitMode::Llvm | EmitMode::Asm => {
            emit_backend_artifact(compilation, emit, output, diagnostics)
        }
    }
}

fn build_error(
    error: impl std::fmt::Display + std::fmt::Debug,
    diagnostics: build_options::DiagnosticsMode,
) -> String {
    if diagnostics.is_internal() {
        format!("{error}\n internal: {error:#?}")
    } else {
        error.to_string()
    }
}

fn emit_backend_artifact(
    compilation: &Compilation,
    emit: EmitMode,
    output: &Path,
    diagnostics: build_options::DiagnosticsMode,
) -> Result<(), String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("severian");
    let working = parent.join(format!(".{stem}.severian"));
    fs::create_dir_all(&working).map_err(|error| error.to_string())?;
    let source_mlir = working.join("source.mlir");
    let llvm_mlir = working.join("llvm.mlir");
    let llvm_ir = working.join("module.ll");
    fs::write(&source_mlir, compilation.mlir.as_str()).map_err(|error| error.to_string())?;
    let result = severian_backend::llvm::lower_module_to_llvm_ir(
        &source_mlir,
        &llvm_mlir,
        &llvm_ir,
        &severian_backend::llvm::LlvmLoweringOptions::native(),
    )
    .map_err(|error| build_error(error, diagnostics))
    .and_then(|_| {
        if emit == EmitMode::Llvm {
            fs::copy(&llvm_ir, output)
                .map(|_| ())
                .map_err(|error| build_error(error, diagnostics))
        } else {
            let clang = severian_backend::toolchain::find_required_tool(
                severian_backend::toolchain::Tool::Clang,
            )
            .map_err(|error| build_error(error, diagnostics))?;
            let status = Command::new(clang)
                .args(["-S", "-x", "ir"])
                .arg(&llvm_ir)
                .arg("-o")
                .arg(output)
                .status()
                .map_err(|error| build_error(error, diagnostics))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("assembly emission failed with {status}"))
            }
        }
    });
    let _ = fs::remove_dir_all(&working);
    result
}

fn resolve_targets(input: &Path) -> Result<Vec<BinaryTarget>, String> {
    if input.is_file() {
        if input.extension().and_then(|value| value.to_str()) != Some("sev") {
            return Err(format!("{} is not a Severian source file", input.display()));
        }
        let manifest = severian_package::find_manifest(input);
        let package_root = manifest
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .or_else(|| input.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        let name = input
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("main")
            .to_owned();
        return Ok(vec![BinaryTarget {
            name,
            source: input.to_owned(),
            package_root,
        }]);
    }
    if !input.is_dir() {
        return Err(format!("{} does not exist", input.display()));
    }
    let manifest = severian_package::nearest_manifest(input);
    if manifest.is_some() || input.join("main.sev").is_file() {
        let targets =
            severian_package::workspace_binary_targets(input).map_err(|error| error.to_string())?;
        if !targets.is_empty() {
            return Ok(targets);
        }
    }
    let mut sources = Vec::new();
    collect_sources(input, &mut sources).map_err(|error| error.to_string())?;
    sources.sort();
    Ok(sources
        .into_iter()
        .map(|source| {
            let name = source
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("main")
                .to_owned();
            BinaryTarget {
                name,
                package_root: input.to_owned(),
                source,
            }
        })
        .collect())
}

fn collect_sources(directory: &Path, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|value| value.to_str());
            if name != Some("target") && !name.is_some_and(|name| name.starts_with('.')) {
                collect_sources(&path, output)?;
            }
        } else if path.extension().and_then(|value| value.to_str()) == Some("sev") {
            output.push(path);
        }
    }
    Ok(())
}

fn fmt_command(args: &[String]) -> Result<(), String> {
    let mut check = false;
    let mut input = PathBuf::from(".");
    for argument in args {
        if argument == "--check" {
            check = true;
        } else if input == Path::new(".") {
            input = PathBuf::from(argument);
        } else {
            return Err("usage: sev fmt [path] [--check]".into());
        }
    }
    let mut sources = if input.is_file() {
        vec![input]
    } else {
        let mut sources = Vec::new();
        collect_sources(&input, &mut sources).map_err(|error| error.to_string())?;
        sources.sort();
        sources
    };
    sources.dedup();
    let mut changed = Vec::new();
    for path in sources {
        let source = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        let tokens = severian_lexer::lex(&source).map_err(|error| error.to_string())?;
        let module = severian_parser::parse(&tokens).map_err(|error| error.to_string())?;
        let formatted = format_contracts(&source, &module);
        if formatted != source {
            changed.push(path.clone());
            if !check {
                fs::write(&path, formatted).map_err(|error| error.to_string())?;
                println!("Formatted {}", path.display());
            }
        }
    }
    if check && !changed.is_empty() {
        return Err(format!(
            "{} file(s) require formatting:\n{}",
            changed.len(),
            changed
                .iter()
                .map(|path| format!("  {}", path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    Ok(())
}

fn format_contracts(source: &str, module: &severian_ast::Module) -> String {
    fn format_with_conditions(conditions: &[String], indent: &str, leading_space: bool) -> String {
        let prefix = if leading_space { " with" } else { "with" };
        if conditions.len() == 1 {
            return format!("{prefix} {{ {} }}", conditions[0]);
        }
        let mut formatted = format!("{prefix}\n{indent}{{\n");
        for condition in conditions {
            formatted.push_str(&format!("{indent}    {condition},\n"));
        }
        formatted.push_str(&format!("{indent}}}"));
        formatted
    }

    fn function_contracts<'a>(
        function: &'a severian_ast::FunctionDecl,
        output: &mut Vec<&'a severian_ast::FunctionContract>,
    ) {
        if let Some(contract) = &function.contract {
            output.push(contract);
        }
        for test in &function.tests {
            if let Some(contract) = &test.contract {
                output.push(contract);
            }
        }
    }

    let mut contracts = Vec::new();
    let mut constrained_fields = Vec::new();
    for item in &module.items {
        match item {
            severian_ast::Item::Function(function) => function_contracts(function, &mut contracts),
            severian_ast::Item::Class(class) => {
                constrained_fields.extend(
                    class
                        .fields
                        .iter()
                        .filter(|field| !field.constraints.is_empty()),
                );
                for constructor in &class.constructors {
                    if let Some(contract) = &constructor.contract {
                        contracts.push(contract);
                    }
                    for test in &constructor.tests {
                        if let Some(contract) = &test.contract {
                            contracts.push(contract);
                        }
                    }
                }
                for method in &class.methods {
                    function_contracts(method, &mut contracts);
                }
            }
            _ => {}
        }
    }
    let mut replacements = Vec::new();
    for contract in contracts {
        let mut replace_start = contract.span.start;
        while replace_start > 0 && source.as_bytes()[replace_start - 1].is_ascii_whitespace() {
            replace_start -= 1;
        }
        if source[..replace_start].ends_with("with") {
            replace_start -= "with".len();
            while replace_start > 0 && matches!(source.as_bytes()[replace_start - 1], b' ' | b'\t')
            {
                replace_start -= 1;
            }
        }
        let header_line = source[..replace_start]
            .rsplit_once('\n')
            .map_or(&source[..replace_start], |(_, line)| line);
        let indent = header_line
            .chars()
            .take_while(|character| character.is_whitespace())
            .collect::<String>();
        let mut conditions = Vec::new();
        for clause in &contract.clauses {
            let text = normalize_contract_clause(&source[clause.span.start..clause.span.end]);
            conditions.push(text);
        }
        for capability in &contract.capabilities {
            conditions.push(format!("with {}", capability.name));
        }
        let mut replacement = format_with_conditions(&conditions, &indent, true);
        let mut replace_end = contract.span.end;
        while matches!(source.as_bytes().get(replace_end), Some(b' ' | b'\t')) {
            replace_end += 1;
        }
        if source.as_bytes().get(replace_end) == Some(&b':') {
            replace_end += 1;
            replacement.push(':');
            while matches!(source.as_bytes().get(replace_end), Some(b' ' | b'\t')) {
                replace_end += 1;
            }
        }
        replacements.push((replace_start, replace_end, replacement));
    }
    for field in constrained_fields {
        let first = field.constraints.first().unwrap().span();
        let last = field.constraints.last().unwrap().span();
        let prefix = &source[field.span.start..first.start];
        let with_offset = prefix
            .rfind("with")
            .expect("constrained field source contains `with`");
        let replace_start = field.span.start + with_offset;
        let mut replace_end = last.end;
        while matches!(
            source.as_bytes().get(replace_end),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            replace_end += 1;
        }
        if source.as_bytes().get(replace_end) == Some(&b',') {
            replace_end += 1;
            while matches!(
                source.as_bytes().get(replace_end),
                Some(b' ' | b'\t' | b'\n' | b'\r')
            ) {
                replace_end += 1;
            }
        }
        if source.as_bytes().get(replace_end) == Some(&b'}') {
            replace_end += 1;
        }
        let field_line = source[..field.span.start]
            .rsplit_once('\n')
            .map_or(&source[..field.span.start], |(_, line)| line);
        let indent = field_line
            .chars()
            .take_while(|character| character.is_whitespace())
            .collect::<String>();
        let conditions = field
            .constraints
            .iter()
            .map(|constraint| {
                let span = constraint.span();
                normalize_contract_clause(&source[span.start..span.end])
            })
            .collect::<Vec<_>>();
        replacements.push((
            replace_start,
            replace_end,
            format_with_conditions(&conditions, &indent, false),
        ));
    }
    replacements.sort_by_key(|(start, _, _)| *start);
    let mut formatted = source.to_owned();
    for (start, end, replacement) in replacements.into_iter().rev() {
        formatted.replace_range(start..end, &replacement);
    }
    formatted
}

fn normalize_contract_clause(source: &str) -> String {
    let characters = source.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut index = 0;
    let mut quote_width = 0;
    let mut escaped = false;
    while index < characters.len() {
        if quote_width == 0 {
            if characters[index].is_whitespace() {
                if !output.is_empty() && !output.ends_with(' ') {
                    output.push(' ');
                }
                index += 1;
                continue;
            }
            if characters[index] == '"' {
                quote_width = if characters.get(index + 1) == Some(&'"')
                    && characters.get(index + 2) == Some(&'"')
                {
                    3
                } else {
                    1
                };
            }
        } else if escaped {
            escaped = false;
        } else if characters[index] == '\\' && quote_width == 1 {
            escaped = true;
        } else if quote_width == 1 && characters[index] == '"' {
            quote_width = 0;
        } else if quote_width == 3
            && characters[index] == '"'
            && characters.get(index + 1) == Some(&'"')
            && characters.get(index + 2) == Some(&'"')
        {
            output.extend(['"', '"', '"']);
            index += 3;
            quote_width = 0;
            continue;
        }
        output.push(characters[index]);
        index += 1;
    }
    output.trim().to_owned()
}

fn build_libraries(source: &Path, built: &mut HashSet<PathBuf>) -> Result<(), String> {
    let Some(manifest) = severian_package::find_manifest(source) else {
        return Ok(());
    };
    for library in
        severian_package::library_build_plan(&manifest).map_err(|error| error.to_string())?
    {
        if !built.insert(library.artifact.clone()) {
            continue;
        }
        compile_dependency_path(&library.source, &library.manifest)
            .map_err(|error| format!("could not build library `{}`: {error}", library.name))?;
        severian_package::write_library_artifact(&library).map_err(|error| error.to_string())?;
        println!("Built {} -> {}", library.name, library.artifact.display());
    }
    Ok(())
}

fn artifact_path(target: &BinaryTarget, emit: EmitMode) -> PathBuf {
    let directory = target.package_root.join("target").join("debug");
    let extension = emit.extension();
    if extension.is_empty() {
        directory.join(&target.name)
    } else if emit == EmitMode::StableHlo {
        directory.join(format!("{}.stablehlo.{extension}", target.name))
    } else {
        directory.join(format!("{}.{extension}", target.name))
    }
}

fn check_targets(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    let mut checked = 0;
    for target in targets {
        check_path(&target.source).map_err(|error| error.to_string())?;
        checked += 1;
        println!("Checked {}", target.source.display());
    }
    let policy = BuildPolicy::for_input(input).map_err(|error| error.to_string())?;
    architecture_command::enforce(&policy)?;
    println!("{checked} checked");
    Ok(())
}

fn lint_command(args: &[String]) -> Result<(), String> {
    let mut input = PathBuf::from(".");
    let mut fix = false;
    let mut has_input = false;
    for argument in args {
        match argument.as_str() {
            "--fix" => fix = true,
            value if !value.starts_with('-') && !has_input => {
                input = PathBuf::from(value);
                has_input = true;
            }
            value => return Err(format!("unknown lint option `{value}`\n{}", usage())),
        }
    }

    let mut sources = if input.is_file() {
        if input.extension().and_then(|value| value.to_str()) != Some("sev") {
            return Err(format!("{} is not a Severian source file", input.display()));
        }
        vec![input]
    } else if input.is_dir() {
        let mut sources = Vec::new();
        collect_sources(&input, &mut sources).map_err(|error| error.to_string())?;
        sources.sort();
        sources
    } else {
        return Err(format!("{} does not exist", input.display()));
    };
    sources.dedup();

    let mut warning_count = 0;
    let mut error_count = 0;
    let mut fixed_count = 0;
    for path in sources {
        let original = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        let tokens = severian_lexer::lex(&original)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let module = severian_parser::parse(&tokens)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let report = severian_diagnostics::naming::check(&module, &tokens, &original, &path);
        let contract_layout_is_canonical = format_contracts(&original, &module) == original;

        let (source, report) = if fix {
            let naming_fixed =
                severian_diagnostics::naming::apply_safe_fixes(&original, &tokens, &report);
            let naming_tokens = severian_lexer::lex(&naming_fixed)
                .map_err(|error| format!("{} after naming fixes: {error}", path.display()))?;
            let naming_module = severian_parser::parse(&naming_tokens)
                .map_err(|error| format!("{} after naming fixes: {error}", path.display()))?;
            let fixed = format_contracts(&naming_fixed, &naming_module);
            if fixed != original {
                fs::write(&path, &fixed).map_err(|error| error.to_string())?;
                fixed_count += 1;
            }
            let fixed_tokens = severian_lexer::lex(&fixed)
                .map_err(|error| format!("{} after fixes: {error}", path.display()))?;
            let fixed_module = severian_parser::parse(&fixed_tokens)
                .map_err(|error| format!("{} after fixes: {error}", path.display()))?;
            let fixed_report =
                severian_diagnostics::naming::check(&fixed_module, &fixed_tokens, &fixed, &path);
            (fixed, fixed_report)
        } else {
            (original, report)
        };

        if !fix && !contract_layout_is_canonical {
            eprintln!(
                "warning[lint::contract-layout]: contract conditions do not use canonical layout\n --> {}\n help: use one inline condition or vertically aligned braces for multiple conditions; run `sev lint --fix`",
                path.display(),
            );
            warning_count += 1;
        }

        if !report.diagnostics.diagnostics().is_empty() {
            let mut source_map = severian_source::SourceMap::new();
            source_map.add(path.clone(), source.clone());
            let rendered = severian_diagnostics::render::render_bag(
                &report.diagnostics,
                Some(&source_map),
                &severian_diagnostics::render::RenderOptions {
                    color: false,
                    ..Default::default()
                },
            );
            eprintln!("{rendered}");
        }
        warning_count += report.diagnostics.warning_count();
        error_count += report.diagnostics.error_count();

        match check_path(&path) {
            Ok(program) => {
                let semantic = severian_diagnostics::lint::run_with_source(
                    &program,
                    &Default::default(),
                    &path,
                    &source,
                );
                if !semantic.diagnostics().is_empty() {
                    let mut source_map = severian_source::SourceMap::new();
                    source_map.add(path.clone(), source.clone());
                    eprintln!(
                        "{}",
                        severian_diagnostics::render::render_bag(
                            &semantic,
                            Some(&source_map),
                            &severian_diagnostics::render::RenderOptions {
                                color: false,
                                ..Default::default()
                            },
                        )
                    );
                }
                warning_count += semantic.warning_count();
                error_count += semantic.error_count();
            }
            // `sev lint` owns lint diagnostics. Syntax and type errors belong to
            // `sev check`/`sev build`, and should not make an otherwise valid
            // naming-lint invocation fail before its safe fixes can be applied.
            Err(_) => {}
        }
    }

    if fix {
        println!(
            "Fixed {fixed_count} file(s); {error_count} error(s) and {warning_count} warning(s) remain"
        );
    } else {
        println!("{error_count} error(s); {warning_count} warning(s)");
    }
    if error_count > 0 {
        return Err(format!("lint failed with {error_count} error(s)"));
    }
    Ok(())
}

fn run_command(args: &[String]) -> Result<(), String> {
    let (input, application_arguments, diagnostics) = runtime_diagnostics::parse_run_args(args)
        .map_err(|error| format!("{error}\n{}", usage()))?;
    run_targets(&input, &application_arguments, diagnostics)
}

fn run_targets(
    input: &Path,
    application_arguments: &[String],
    diagnostics_override: Option<build_options::DiagnosticsMode>,
) -> Result<(), String> {
    let diagnostics = diagnostics_override.unwrap_or(build_options::load(input)?.diagnostics);
    let targets = resolve_targets(input)?;
    if targets.is_empty() {
        return Err(format!(
            "no runnable Severian targets found under {}",
            input.display()
        ));
    }
    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        let output = artifact_path(&target, EmitMode::Executable);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        if compilation.hir.main().is_some() {
            runtime_diagnostics::compile_executable(&compilation, &output, diagnostics)?;
        } else if native_test_count(&compilation.hir) > 0 {
            compile_native_tests(&compilation, &output).map_err(|error| error.to_string())?;
        } else {
            for module in emit_non_executable_module(&compilation, &target)? {
                println!(
                    "Compiled {} function(s) -> {}",
                    compilation.hir.functions.len(),
                    module.display()
                );
            }
            continue;
        }
        let mut command = Command::new(&output);
        command.args(application_arguments);
        let has_xla_regions = compilation.optimized_hir.functions.iter().any(|function| {
            function
                .decorators
                .iter()
                .any(|decorator| decorator.package == "tensor")
        });
        if has_xla_regions && std::env::var_os("SEVERIAN_ROCM_PJRT_PLUGIN").is_none() {
            if let Some(plugin) = local_rocm_pjrt_plugin(&target.package_root) {
                command.env("SEVERIAN_ROCM_PJRT_PLUGIN", plugin);
            }
        }
        let report = runtime_diagnostics::report_path(&output);
        command.env(runtime_diagnostics::REPORT_ENV, &report);
        let status = command
            .status()
            .map_err(|error| format!("could not run {}: {error}", output.display()))?;
        if !status.success() {
            if let Some(rendered) = runtime_diagnostics::take_report(&report, diagnostics, &output)?
            {
                return Err(rendered);
            }
            return Err(runtime_diagnostics::signal_fallback(
                status,
                &output,
                diagnostics,
            ));
        }
        let _ = fs::remove_file(report);
    }
    Ok(())
}

fn debug_targets(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    if targets.is_empty() {
        return Err(format!(
            "no debuggable Severian targets found under {}",
            input.display()
        ));
    }
    let configured = std::env::var_os("SEVERIAN_DEBUGGER").map(PathBuf::from);
    let debugger = configured
        .or_else(|| find_command(&["lldb", "gdb"]))
        .ok_or_else(|| {
            "no debugger found; install `lldb` or `gdb`, or set SEVERIAN_DEBUGGER".to_string()
        })?;
    let debugger_name = debugger
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("debugger");

    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        if compilation.hir.main().is_none() {
            return Err(format!(
                "{} has no `main()` function to debug",
                target.source.display()
            ));
        }
        let output = artifact_path(&target, EmitMode::Executable);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let options = severian_backend::NativeCompileOptions {
            debug: true,
            ..severian_backend::NativeCompileOptions::default()
        };
        compile_native_with_options(&compilation, &output, &options)
            .map_err(|error| error.to_string())?;
        println!("Debug build {} -> {}", target.name, output.display());

        let mut command = Command::new(&debugger);
        if debugger_name.contains("lldb") {
            command.arg("--").arg(&output);
        } else if debugger_name.contains("gdb") {
            command.arg("--args").arg(&output);
        } else {
            command.arg(&output);
        }
        let status = command
            .status()
            .map_err(|error| format!("could not launch {}: {error}", debugger.display()))?;
        if !status.success() {
            return Err(format!("{} exited with {status}", debugger.display()));
        }
    }
    Ok(())
}

fn emit_non_executable_module(
    compilation: &Compilation,
    target: &BinaryTarget,
) -> Result<Vec<PathBuf>, String> {
    let kernels = severian_lowering::kernel::collect(&compilation.mir);
    if !kernels.is_empty() {
        use severian_lowering::kernel::{
            emit_stablehlo, emit_triton_ir, select_backend, KernelBackend, KernelTarget,
        };
        let mut outputs = Vec::new();
        for kernel in kernels {
            let selection = select_backend(&kernel, kernel.policy, KernelTarget::Gpu)
                .map_err(|error| error.to_string())?;
            let (artifact, extension) = match selection.selected {
                KernelBackend::Triton => (
                    emit_triton_ir(&kernel).map_err(|error| error.to_string())?,
                    "ttir",
                ),
                KernelBackend::Xla => (
                    emit_stablehlo(&kernel)
                        .map_err(|error| error.to_string())?
                        .as_str()
                        .to_owned(),
                    "stablehlo.mlir",
                ),
                KernelBackend::Auto => unreachable!("backend selection resolves auto"),
                KernelBackend::Llvm => unreachable!("GPU selection does not choose LLVM"),
            };
            let output = target
                .package_root
                .join("target/debug")
                .join(format!("{}.{extension}", kernel.name));
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            fs::write(&output, artifact).map_err(|error| error.to_string())?;
            outputs.push(output);
        }
        return Ok(outputs);
    }
    let output = artifact_path(target, EmitMode::Llvm);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    emit_backend_artifact(
        compilation,
        EmitMode::Llvm,
        &output,
        build_options::DiagnosticsMode::User,
    )?;
    Ok(vec![output])
}

fn local_rocm_pjrt_plugin(start: &Path) -> Option<PathBuf> {
    for directory in start.ancestors() {
        let python_lib = directory.join(".venv/lib");
        let Ok(versions) = fs::read_dir(python_lib) else {
            continue;
        };
        for version in versions.flatten() {
            let candidate = version
                .path()
                .join("site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn test_targets(input: &Path, profile_only: bool) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    let mut total = 0;
    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        let count = if profile_only {
            native_profile_test_count(&compilation.hir)
        } else {
            native_test_count(&compilation.hir)
        };
        if count == 0 {
            continue;
        }
        let output = target
            .package_root
            .join("target")
            .join("debug")
            .join(format!("{}-tests", target.name));
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        if profile_only {
            compile_native_profile_tests(&compilation, &output)
        } else {
            compile_native_tests(&compilation, &output)
        }
        .map_err(|error| error.to_string())?;
        let status = Command::new(&output)
            .status()
            .map_err(|error| format!("could not run {}: {error}", output.display()))?;
        if !status.success() {
            return Err(format!(
                "native tests for {} failed with {status}",
                target.source.display()
            ));
        }
        total += count;
    }
    if profile_only {
        println!("{total} profile test(s) passed");
    } else {
        println!("{total} passed");
    }
    Ok(())
}

fn integration_test_targets(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    let mut total = 0;
    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        let count = native_integration_test_count(&compilation.hir);
        if count == 0 {
            continue;
        }
        let output = target
            .package_root
            .join("target")
            .join("debug")
            .join(format!("{}-integration-tests", target.name));
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        compile_native_integration_tests(&compilation, &output)
            .map_err(|error| error.to_string())?;
        let status = Command::new(&output)
            .status()
            .map_err(|error| format!("could not run {}: {error}", output.display()))?;
        if !status.success() {
            return Err(format!(
                "integration tests for {} failed with {status}",
                target.source.display()
            ));
        }
        total += count;
    }
    println!("{total} integration test(s) passed");
    Ok(())
}

fn test_command(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut mutate = false;
    let mut profile = false;
    let mut memory = false;
    let mut leaks = false;
    let mut sanitizers = Vec::new();
    let mut limit = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--mutate" => {
                mutate = true;
                index += 1;
            }
            "--profile" => {
                profile = true;
                index += 1;
            }
            "--memory" => {
                memory = true;
                index += 1;
            }
            "--leaks" => {
                memory = true;
                leaks = true;
                index += 1;
            }
            "--sanitizer" if index + 1 < args.len() => {
                memory = true;
                let sanitizer = parse_native_sanitizer(&args[index + 1])?;
                if !sanitizers.contains(&sanitizer) {
                    sanitizers.push(sanitizer);
                }
                index += 2;
            }
            "--limit" if index + 1 < args.len() => {
                limit = Some(
                    args[index + 1]
                        .parse::<usize>()
                        .map_err(|_| "--limit must be a positive integer".to_string())?,
                );
                index += 2;
            }
            value if !value.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(value));
                index += 1;
            }
            value => return Err(format!("unknown test option `{value}`\n{}", usage())),
        }
    }
    let input = input.unwrap_or_else(|| PathBuf::from("."));
    if mutate && (profile || memory) {
        Err("`--mutate` cannot be combined with `--profile` or `--memory`".into())
    } else if mutate {
        mutation_test_targets(&input, limit)
    } else if memory {
        memory_test_targets(&input, profile, sanitizers, leaks)
    } else {
        test_targets(&input, profile)
    }
}

fn mutation_test_targets(input: &Path, limit: Option<usize>) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    let mut generated = 0;
    let mut killed = 0;
    let mut survived = 0;
    let mut invalid = 0;

    for target in targets {
        let compilation = compile_path(&target.source)
            .map_err(|error| format!("{}: {error}", target.source.display()))?;
        if native_test_count(&compilation.hir) == 0 {
            continue;
        }

        let (baseline, _) = native_test_compilation(&compilation)
            .map_err(|error| format!("{}: {error}", target.source.display()))?;
        let directory = target.package_root.join("target").join("mutants");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let baseline_binary = directory.join(format!("{}-baseline", target.name));
        compile_native(&baseline, &baseline_binary).map_err(|error| error.to_string())?;
        if run_with_timeout(&baseline_binary, 10)?.is_failure() {
            return Err(format!(
                "baseline tests for {} do not pass; mutation testing requires a green baseline",
                target.source.display()
            ));
        }

        let available = severian_driver::mutation::count(&compilation);
        let selected = limit.map_or(available, |limit| limit.min(available));
        for mutation_index in 0..selected {
            let Some((mutated, mutation)) =
                severian_driver::mutation::apply(&compilation, mutation_index)
                    .map_err(|error| format!("{}: {error}", target.source.display()))?
            else {
                continue;
            };
            generated += 1;
            let (runnable, _) = native_test_compilation(&mutated)
                .map_err(|error| format!("{}: {error}", target.source.display()))?;
            let binary = directory.join(format!("{}-{mutation_index}", target.name));
            if let Err(error) = compile_native(&runnable, &binary) {
                invalid += 1;
                println!(
                    "INVALID  {}: {} ({error})",
                    mutation_location(&mutation),
                    mutation.description
                );
                continue;
            }
            match run_with_timeout(&binary, 10)? {
                MutantStatus::Passed => {
                    survived += 1;
                    println!(
                        "SURVIVED {}: {}",
                        mutation_location(&mutation),
                        mutation.description
                    );
                }
                MutantStatus::Failed => {
                    killed += 1;
                    println!(
                        "KILLED   {}: {}",
                        mutation_location(&mutation),
                        mutation.description
                    );
                }
                MutantStatus::TimedOut => {
                    killed += 1;
                    println!(
                        "KILLED   {}: {} (timed out)",
                        mutation_location(&mutation),
                        mutation.description
                    );
                }
            }
        }
    }

    let viable = killed + survived;
    let score = if viable == 0 {
        100.0
    } else {
        killed as f64 * 100.0 / viable as f64
    };
    println!(
        "Mutants generated: {generated}\nKilled:            {killed}\nSurvived:          {survived}\nInvalid:           {invalid}\nMutation score:    {score:.1}%"
    );
    Ok(())
}

fn mutation_location(mutation: &severian_driver::mutation::Mutation) -> String {
    match (&mutation.file, mutation.line) {
        (Some(file), Some(line)) => format!("{}:{line}", file.display()),
        (Some(file), None) => file.display().to_string(),
        _ => format!("mutant {}", mutation.index),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MutantStatus {
    Passed,
    Failed,
    TimedOut,
}

impl MutantStatus {
    fn is_failure(self) -> bool {
        self != Self::Passed
    }
}

fn run_with_timeout(binary: &Path, seconds: u64) -> Result<MutantStatus, String> {
    let mut child = Command::new(binary)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|error| format!("could not run {}: {error}", binary.display()))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Ok(if status.success() {
                MutantStatus::Passed
            } else {
                MutantStatus::Failed
            });
        }
        if std::time::Instant::now() >= deadline {
            child.kill().map_err(|error| error.to_string())?;
            let _ = child.wait();
            return Ok(MutantStatus::TimedOut);
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn coverage(input: &Path) -> Result<(), String> {
    let policy = BuildPolicy::for_input(input).map_err(|error| error.to_string())?;
    coverage_with_policy(input, &policy)
}

fn coverage_with_policy(input: &Path, policy: &BuildPolicy) -> Result<(), String> {
    let targets = resolve_targets(input)?
        .into_iter()
        .filter(|target| !is_expected_negative_coverage_fixture(&target.source))
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return Err(format!(
            "no Severian targets found under {}",
            input.display()
        ));
    }
    let report_root = targets[0].package_root.join("target").join("coverage");
    fs::create_dir_all(&report_root).map_err(|error| error.to_string())?;
    let mut all_regions = severian_coverage::CoverageSourceMap::default();
    let mut all_hits = std::collections::BTreeSet::new();
    let mut executed = 0;
    let mut tests = 0;
    let mut failures = Vec::new();

    for (index, target) in targets.iter().enumerate() {
        let compilation = match compile_path(&target.source) {
            Ok(compilation) => compilation,
            Err(error) => {
                failures.push(format!("{}: {error}", target.source.display()));
                continue;
            }
        };
        let declared_count = native_test_count(&compilation.hir);
        let (_, source_regions) = match severian_driver::coverage::instrument(&compilation) {
            Ok(instrumented) => instrumented,
            Err(error) => {
                failures.push(format!("{}: {error}", target.source.display()));
                continue;
            }
        };
        all_regions.extend(source_regions);
        let (runnable, count) = if declared_count > 0 {
            match native_coverage_test_compilation(&compilation) {
                Ok(runnable) => runnable,
                Err(error) => {
                    failures.push(format!("{}: {error}", target.source.display()));
                    continue;
                }
            }
        } else {
            (compilation, 0)
        };
        let (instrumented, _) = match severian_driver::coverage::instrument(&runnable) {
            Ok(instrumented) => instrumented,
            Err(error) => {
                failures.push(format!("{}: {error}", target.source.display()));
                continue;
            }
        };
        if count == 0 {
            continue;
        }

        let stem = format!("{}-{index}", target.name);
        let binary = report_root.join(format!("{stem}-tests"));
        let hits = report_root.join(format!("{stem}.hits"));
        if hits.exists() {
            fs::remove_file(&hits).map_err(|error| error.to_string())?;
        }
        if let Err(error) = compile_native(&instrumented, &binary) {
            failures.push(format!("{}: {error}", target.source.display()));
            continue;
        }
        let output = Command::new(&binary)
            .env("SEVERIAN_COVERAGE_FILE", &hits)
            .output()
            .map_err(|error| format!("could not run {}: {error}", binary.display()))?;
        if !output.status.success() {
            failures.push(format!(
                "{}: tests failed with {}: {}",
                target.source.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
            continue;
        }
        all_hits.extend(
            severian_coverage::read_language_hits(&hits).map_err(|error| error.to_string())?,
        );
        tests += count;
        executed += 1;
    }

    let map_path = report_root.join("coverage-map.json");
    all_regions
        .save_json(&map_path)
        .map_err(|error| error.to_string())?;
    let hits_path = report_root.join("coverage.hits");
    severian_coverage::save_language_hits(&hits_path, &all_hits)
        .map_err(|error| error.to_string())?;
    let project_regions = all_regions.within_root(&targets[0].package_root);
    let (report, files) = severian_coverage::language_report(&project_regions, &all_hits);
    let report_path = report_root.join("coverage-report.json");
    severian_coverage::save_language_report(&report_path, &report, &files)
        .map_err(|error| error.to_string())?;
    print!("{}", severian_coverage::render_files(&files));
    print!("{}", severian_coverage::report::render_text(&report));
    println!(
        "Executed {tests} test(s) across {executed} target(s); {} failure(s); report: {}; map: {}",
        failures.len(),
        report_path.display(),
        map_path.display()
    );
    for failure in &failures {
        eprintln!("UNCOVERED {failure}");
    }
    if !failures.is_empty() {
        return Err(format!(
            "coverage could not compile or execute {} target(s)",
            failures.len()
        ));
    }

    coverage_policy::enforce(&report, &files, &policy.coverage)
}

fn is_expected_negative_coverage_fixture(source: &Path) -> bool {
    source.file_name().and_then(|name| name.to_str()) == Some("invalid.sev")
        && source
            .components()
            .any(|component| component.as_os_str() == "bugs")
}

fn memory_command(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut sanitizers = Vec::new();
    let mut leaks = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--leaks" => {
                leaks = true;
                index += 1;
            }
            "--sanitizer" if index + 1 < args.len() => {
                let sanitizer = parse_native_sanitizer(&args[index + 1])?;
                if !sanitizers.contains(&sanitizer) {
                    sanitizers.push(sanitizer);
                }
                index += 2;
            }
            value if !value.starts_with('-') && input.is_none() => {
                input = Some(PathBuf::from(value));
                index += 1;
            }
            value => return Err(format!("unknown memory option `{value}`\n{}", usage())),
        }
    }
    let input = input.ok_or_else(usage)?;
    memory_test_targets(&input, false, sanitizers, leaks)
}

fn parse_native_sanitizer(value: &str) -> Result<NativeSanitizer, String> {
    match value {
        "address" => Ok(NativeSanitizer::Address),
        "thread" => Ok(NativeSanitizer::Thread),
        "memory" => Ok(NativeSanitizer::Memory),
        "undefined" => Ok(NativeSanitizer::Undefined),
        value => Err(format!("unknown sanitizer `{value}`")),
    }
}

fn memory_test_targets(
    input: &Path,
    profile_only: bool,
    mut sanitizers: Vec<NativeSanitizer>,
    leaks: bool,
) -> Result<(), String> {
    if sanitizers.is_empty() {
        sanitizers.extend([NativeSanitizer::Address, NativeSanitizer::Undefined]);
    }
    let has_thread = sanitizers.contains(&NativeSanitizer::Thread);
    let has_memory = sanitizers.contains(&NativeSanitizer::Memory);
    if (has_thread || has_memory) && sanitizers.len() > 1 {
        return Err(
            "thread and memory sanitizers must run alone; address may be combined with undefined"
                .into(),
        );
    }
    if leaks && !sanitizers.contains(&NativeSanitizer::Address) {
        return Err("`--leaks` requires the address sanitizer".into());
    }

    let options = NativeCompileOptions {
        sanitizers,
        ..NativeCompileOptions::default()
    };
    let targets = resolve_targets(input)?;
    let mut executed = 0;
    let mut findings = 0;
    let mut tool_failures = 0;
    for target in targets {
        let compilation = compile_path(&target.source)
            .map_err(|error| format!("{}: {error}", target.source.display()))?;
        let count = if profile_only {
            native_profile_test_count(&compilation.hir)
        } else {
            native_test_count(&compilation.hir)
        };
        if count == 0 {
            continue;
        }
        let (runnable, count) = if profile_only {
            native_profile_test_compilation(&compilation)
        } else {
            native_test_compilation(&compilation)
        }
        .map_err(|error| format!("{}: {error}", target.source.display()))?;
        let directory = target.package_root.join("target").join("memory");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let suffix = if profile_only {
            "profile-tests"
        } else {
            "tests"
        };
        let binary = directory.join(format!("{}-{suffix}", target.name));
        compile_native_with_options(&runnable, &binary, &options)
            .map_err(|error| format!("{}: {error}", target.source.display()))?;
        let label = if profile_only {
            "Memory + profile checking"
        } else {
            "Memory checking"
        };
        println!("{label} {} ({count} test(s))", target.source.display());
        let output = Command::new(&binary)
            .env(
                "ASAN_OPTIONS",
                format!(
                    "detect_leaks={}:halt_on_error=1:abort_on_error=1",
                    usize::from(leaks)
                ),
            )
            .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
            .env("MSAN_OPTIONS", "halt_on_error=1")
            .env("TSAN_OPTIONS", "halt_on_error=1")
            .output()
            .map_err(|error| format!("could not run {}: {error}", binary.display()))?;
        print!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        executed += 1;
        if output.status.success() {
            println!("PASS {}", target.source.display());
        } else if String::from_utf8_lossy(&output.stderr)
            .contains("LeakSanitizer has encountered a fatal error")
        {
            tool_failures += 1;
            println!(
                "UNAVAILABLE {}: LeakSanitizer could not inspect this process",
                target.source.display()
            );
        } else {
            findings += 1;
            println!(
                "FINDING {} exited with {}",
                target.source.display(),
                output.status
            );
        }
    }
    let label = if profile_only {
        "Memory + profile summary"
    } else {
        "Memory summary"
    };
    println!("{label}: {executed} target(s), {findings} target(s) with findings, {tool_failures} tool failure(s)");
    if findings == 0 && tool_failures == 0 {
        Ok(())
    } else {
        Err(format!(
            "memory checking found failures in {findings} target(s) and could not inspect {tool_failures} target(s)"
        ))
    }
}

fn clean(input: Option<&Path>) -> Result<(), String> {
    let input = input.unwrap_or_else(|| Path::new("."));
    let root = if input.is_file() {
        severian_package::find_manifest(input)
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .or_else(|| input.parent().map(Path::to_path_buf))
    } else if input.is_dir() {
        severian_package::nearest_manifest(input)
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .or_else(|| (input != Path::new(".")).then(|| input.to_path_buf()))
    } else {
        None
    }
    .ok_or_else(|| {
        "sev clean needs an explicit Severian source/project outside a manifest workspace"
            .to_string()
    })?;
    let target = root.join("target");
    if target.exists() {
        fs::remove_dir_all(&target).map_err(|error| error.to_string())?;
    }
    println!("Removed {}", target.display());
    Ok(())
}

fn tree(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    for target in targets {
        println!("{}", target.name);
        if let Some(manifest) = severian_package::find_manifest(&target.source) {
            let libraries = severian_package::library_build_plan(&manifest)
                .map_err(|error| error.to_string())?;
            for (index, library) in libraries.iter().enumerate() {
                let branch = if index + 1 == libraries.len() {
                    "└──"
                } else {
                    "├──"
                };
                println!("{branch} {} ({})", library.name, library.manifest.display());
            }
        }
    }
    Ok(())
}

fn metadata(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    println!(
        "{{\n  \"compiler\": \"severian {}\",\n  \"targets\": [",
        env!("CARGO_PKG_VERSION")
    );
    for (index, target) in targets.iter().enumerate() {
        let dependencies = severian_package::find_manifest(&target.source)
            .map(|manifest| severian_package::library_build_plan(&manifest))
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or_default();
        println!("    {{");
        println!("      \"name\": \"{}\",", json_escape(&target.name));
        println!(
            "      \"entry_point\": \"{}\",",
            json_escape(&target.source.display().to_string())
        );
        println!(
            "      \"source_root\": \"{}\",",
            json_escape(
                &target
                    .source
                    .parent()
                    .unwrap_or(Path::new("."))
                    .display()
                    .to_string()
            )
        );
        println!("      \"target\": \"native\",");
        println!(
            "      \"artifact\": \"{}\",",
            json_escape(
                &artifact_path(target, EmitMode::Executable)
                    .display()
                    .to_string()
            )
        );
        print!("      \"dependencies\": [");
        for (dependency_index, dependency) in dependencies.iter().enumerate() {
            if dependency_index > 0 {
                print!(", ");
            }
            print!("\"{}\"", json_escape(&dependency.name));
        }
        println!(
            "]\n    }}{}",
            if index + 1 == targets.len() { "" } else { "," }
        );
    }
    println!("  ]\n}}");
    Ok(())
}

fn explain(code: &str) -> Result<(), String> {
    let explanation = severian_diagnostics::explain::explain(code)
        .ok_or_else(|| format!("no explanation is registered for diagnostic `{code}`"))?;
    println!(
        "{}: {}\n\n{}",
        explanation.code, explanation.title, explanation.text
    );
    Ok(())
}

fn emit_stdout(input: &Path, mode: EmitMode) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    if targets.len() != 1 {
        return Err("stdout emission requires exactly one source target".into());
    }
    let compilation = compile_path(&targets[0].source).map_err(|error| error.to_string())?;
    match mode {
        EmitMode::Mlir => print!("{}", compilation.mlir),
        _ => unreachable!(),
    }
    Ok(())
}

fn legacy_compile(args: &[String]) -> Result<(), String> {
    let output = match args {
        [_, _, flag, output] if flag == "-o" => PathBuf::from(output),
        [_, _] => PathBuf::from("a.out"),
        _ => return Err(usage()),
    };
    let compilation = compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
    compile_native(&compilation, &output).map_err(|error| error.to_string())?;
    println!("{}", output.display());
    Ok(())
}

fn doctor() -> Result<(), String> {
    println!("Severian doctor");
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    status(
        "Severian compiler",
        "PASS",
        &executable.display().to_string(),
    );
    check_command("Rust compiler", &["rustc"], false);
    check_command("Cargo", &["cargo"], false);

    let report = inspect_toolchain();
    for tool in &report.tools {
        match (&tool.path, &tool.version, tool.compatible, tool.required) {
            (Some(path), Some(version), true, _) => status(
                tool.name,
                "PASS",
                &format!("{} — {version}", path.display()),
            ),
            (Some(path), Some(version), false, true) => status(
                tool.name,
                "ERROR",
                &format!("{} — unsupported {version}", path.display()),
            ),
            (Some(path), _, false, false) => status(
                tool.name,
                "WARNING",
                &format!("{} — version unavailable", path.display()),
            ),
            (None, _, _, true) => status(
                tool.name,
                "NOT INSTALLED",
                "required for native compilation",
            ),
            (None, _, _, false) => status(tool.name, "OPTIONAL", "not installed"),
            (Some(path), None, _, true) => status(
                tool.name,
                "ERROR",
                &format!("{} — version unavailable", path.display()),
            ),
            (Some(path), None, _, false) => status(
                tool.name,
                "WARNING",
                &format!("{} — version unavailable", path.display()),
            ),
        }
    }
    check_command(
        "LLVM coverage",
        &["llvm-profdata-21", "llvm-profdata"],
        true,
    );
    check_command("LLVM coverage report", &["llvm-cov-21", "llvm-cov"], true);
    check_command(
        "StableHLO tools",
        &["stablehlo-opt", "stablehlo-translate"],
        true,
    );
    match std::env::var_os("SEVERIAN_PJRT_PLUGIN") {
        Some(path) if Path::new(&path).is_file() => status(
            "PJRT plugin",
            "PASS",
            &Path::new(&path).display().to_string(),
        ),
        Some(path) => status(
            "PJRT plugin",
            "WARNING",
            &format!(
                "configured path {} does not exist",
                Path::new(&path).display()
            ),
        ),
        None => status(
            "PJRT plugin",
            "OPTIONAL",
            "NOT CONFIGURED; native builds are unaffected",
        ),
    }
    status(
        "Host",
        "PASS",
        &format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
    );
    let directory = std::env::current_dir()
        .map_err(|error| error.to_string())?
        .join("target");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("build directory is not writable: {error}"))?;
    let probe = directory.join(".severian-doctor-write-probe");
    fs::write(&probe, b"ok")
        .map_err(|error| format!("build directory is not writable: {error}"))?;
    fs::remove_file(&probe).map_err(|error| error.to_string())?;
    status(
        "Build/cache directory",
        "PASS",
        &directory.display().to_string(),
    );
    if report.sqlite_available {
        status("SQLite", "PASS", "optional runtime available");
    } else {
        status("SQLite", "OPTIONAL", "NOT INSTALLED");
    }
    if report.native_ready() {
        status("Native compilation", "PASS", "ready");
        Ok(())
    } else {
        Err("native compilation is not ready".into())
    }
}

fn check_command(label: &str, candidates: &[&str], optional: bool) {
    match find_command(candidates) {
        Some(path) => status(label, "PASS", &path.display().to_string()),
        None if optional => status(label, "OPTIONAL", "NOT INSTALLED"),
        None => status(
            label,
            "WARNING",
            "NOT INSTALLED; only needed to rebuild Severian",
        ),
    }
}

fn find_command(candidates: &[&str]) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        for candidate in candidates {
            let executable = directory.join(candidate);
            if executable.is_file() {
                return Some(executable);
            }
        }
    }
    None
}

fn status(component: &str, state: &str, detail: &str) {
    println!("{component:<28} {state:<14} {detail}");
}

fn usage() -> String {
    [
        "usage: sev <command> [project-or-source] [options]",
        "  doctor                         diagnose native and optional toolchains",
        "  new <path>                     create a project with package.toml and sev.lock",
        "  init [path]                    initialize a project in an existing directory",
        "  add <package> [--version V] [--path P] add a dependency to package.toml",
        "  remove <package>                remove a dependency and refresh sev.lock",
        "  update [path]                   resolve, verify, cache, and lock dependencies",
        "  publish [path]                  publish an immutable version to the configured registry",
        "  install [--dry-run] [--locked] resolve, approve, and install declared external requirements",
        "  install <package> [--version V] install a registry package binary",
        "  trust list|show <publisher>    inspect the compiler-owned publisher trust registry",
        "  verify [path]                  verify locked packages, trust, signatures, and installed artifacts",
        "  check [path]                   parse, resolve, typecheck, and check ownership",
        "  architecture [path] [--graph] analyze package dependency and layer boundaries",
        "  lint [path] [--fix]            enforce source naming and compatibility style",
        "  fmt [path] [--check]           format contracts and verify canonical layout",
        "  build [path] [--emit KIND] [--target native|xla] [--diagnostics user|internal] [--verify-each] [--max-errors N] [--message-format text|json]",
        "  run [path] [--diagnostics user|internal] [-- args...] build and run native code with application arguments",
        "  test [path]                    build and run native Severian tests",
        "  test [path] --profile          run only profile tests and enforce profile contracts",
        "  test [path] --memory [--leaks] run tests with native memory diagnostics",
        "  test [path] --profile --memory report speed/allocations with memory diagnostics",
        "  debug [path]                   build with debug symbols and launch lldb or gdb",
        "  test [path] --mutate [--limit N] run deterministic mutation testing",
        "  coverage <path>                run tests and report Severian source coverage",
        "  memory <path> [--sanitizer KIND] [--leaks] run native memory diagnostics",
        "  kernel inspect|emit <source>   explain backend choice or emit a standalone kernel",
        "  --emit <stage> <path>          emit hir, mir, mlir, stablehlo, llvm, or asm",
        "  clean [path]                   remove only the Severian project target directory",
        "  tree <path>                    print the Severian package dependency graph",
        "  metadata <path>                print Severian project metadata as JSON",
        "  explain <diagnostic-code>      explain a registered diagnostic",
        "  errors                         list every registered compiler error",
        "  emit kinds: hir, mir, mlir, stablehlo, llvm, asm",
    ]
    .join("\n")
}
