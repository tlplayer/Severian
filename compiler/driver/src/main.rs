use severian_driver::{
    check_path, compile_native, compile_native_tests, compile_path, inspect_toolchain, Compilation,
};
use severian_package::BinaryTarget;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if let Err(error) = execute(std::env::args().skip(1).collect()) {
        eprintln!("error: {error}");
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
        return run_targets(Path::new(command));
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
        "check" if args.len() == 2 => check_targets(Path::new(&args[1])),
        "build" => build_command(&args[1..]).map(|_| ()),
        "run" if args.len() == 2 => run_targets(Path::new(&args[1])),
        "test" if args.len() == 2 => test_targets(Path::new(&args[1])),
        "coverage" if args.len() == 2 => coverage(Path::new(&args[1])),
        "clean" if args.len() <= 2 => clean(args.get(1).map(Path::new)),
        "tree" if args.len() == 2 => tree(Path::new(&args[1])),
        "metadata" if args.len() == 2 => metadata(Path::new(&args[1])),
        "explain" if args.len() == 2 => explain(&args[1]),
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

fn build_command(args: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut input = PathBuf::from(".");
    let mut emit = EmitMode::Executable;
    let mut target = BuildTarget::Native;
    let mut index = 0;
    if args.first().is_some_and(|value| !value.starts_with('-')) {
        input = PathBuf::from(&args[0]);
        index = 1;
    }
    while index < args.len() {
        match args[index].as_str() {
            "--emit" if index + 1 < args.len() => {
                emit = EmitMode::parse(&args[index + 1])?;
                index += 2;
            }
            "--target" if index + 1 < args.len() => {
                target = match args[index + 1].as_str() {
                    "native" | "cpu" => BuildTarget::Native,
                    value if value == "xla" || value.starts_with("xla:") => BuildTarget::Xla,
                    value => {
                        return Err(format!(
                            "unsupported build target `{value}`; use native or xla"
                        ))
                    }
                };
                index += 2;
            }
            value => return Err(format!("unknown build option `{value}`\n{}", usage())),
        }
    }
    if target == BuildTarget::Native && emit == EmitMode::StableHlo {
        return Err("StableHLO emission requires `--target xla`".into());
    }

    let targets = resolve_targets(&input)?;
    if targets.is_empty() {
        return Err(format!(
            "no runnable Severian targets found under {}",
            input.display()
        ));
    }
    let mut libraries = HashSet::new();
    let mut artifacts = Vec::new();
    for target_spec in targets {
        build_libraries(&target_spec.source, &mut libraries)?;
        let compilation = compile_path(&target_spec.source).map_err(|error| error.to_string())?;
        let output = artifact_path(&target_spec, emit);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        emit_artifact(&compilation, emit, target, &output)?;
        println!("Built {} -> {}", target_spec.name, output.display());
        artifacts.push(output);
    }
    Ok(artifacts)
}

fn emit_artifact(
    compilation: &Compilation,
    emit: EmitMode,
    target: BuildTarget,
    output: &Path,
) -> Result<(), String> {
    match emit {
        EmitMode::Executable => {
            compile_native(compilation, output).map_err(|error| error.to_string())
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
                .map_err(|error| error.to_string())?;
            fs::write(output, module.as_str()).map_err(|error| error.to_string())
        }
        EmitMode::Llvm | EmitMode::Asm => emit_backend_artifact(compilation, emit, output),
    }
}

fn emit_backend_artifact(
    compilation: &Compilation,
    emit: EmitMode,
    output: &Path,
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
    .map_err(|error| error.to_string())
    .and_then(|_| {
        if emit == EmitMode::Llvm {
            fs::copy(&llvm_ir, output)
                .map(|_| ())
                .map_err(|error| error.to_string())
        } else {
            let clang = severian_backend::toolchain::find_required_tool(
                severian_backend::toolchain::Tool::Clang,
            )
            .map_err(|error| error.to_string())?;
            let status = Command::new(clang)
                .args(["-S", "-x", "ir"])
                .arg(&llvm_ir)
                .arg("-o")
                .arg(output)
                .status()
                .map_err(|error| error.to_string())?;
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
        let package_root = severian_package::find_manifest(input)
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
    if severian_package::nearest_manifest(input).is_some() || input.join("main.sev").is_file() {
        return severian_package::workspace_binary_targets(input)
            .map_err(|error| error.to_string());
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
            if path.file_name().and_then(|value| value.to_str()) != Some("target") {
                collect_sources(&path, output)?;
            }
        } else if path.extension().and_then(|value| value.to_str()) == Some("sev") {
            output.push(path);
        }
    }
    Ok(())
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
        compile_path(&library.source)
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
        check_path(&target.source)
            .map_err(|error| format!("{}: {error}", target.source.display()))?;
        checked += 1;
        println!("Checked {}", target.source.display());
    }
    println!("{checked} checked");
    Ok(())
}

fn run_targets(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    if targets.is_empty() {
        return Err(format!(
            "no runnable Severian targets found under {}",
            input.display()
        ));
    }
    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        if compilation.hir.main().is_none() {
            return Err(format!("{} has no main function", target.source.display()));
        }
        let output = artifact_path(&target, EmitMode::Executable);
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        compile_native(&compilation, &output).map_err(|error| error.to_string())?;
        let mut command = Command::new(&output);
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
        let status = command
            .status()
            .map_err(|error| format!("could not run {}: {error}", output.display()))?;
        if !status.success() {
            return Err(format!("{} exited with {status}", output.display()));
        }
    }
    Ok(())
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

fn test_targets(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    let mut total = 0;
    for target in targets {
        let compilation = compile_path(&target.source).map_err(|error| error.to_string())?;
        let count = compilation.hir.test_count();
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
        compile_native_tests(&compilation, &output).map_err(|error| error.to_string())?;
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
    println!("{total} passed");
    Ok(())
}

fn coverage(input: &Path) -> Result<(), String> {
    let targets = resolve_targets(input)?;
    for target in &targets {
        compile_path(&target.source).map_err(|error| error.to_string())?;
    }
    Err(format!(
        "coverage is not yet available: {} target(s) checked, but HIR does not retain the stable Severian source spans required for truthful LLVM coverage mapping",
        targets.len()
    ))
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

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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
        "  check <path>                   parse, resolve, typecheck, and check ownership",
        "  build [path] [--emit KIND] [--target native|xla]",
        "  run <path>                     build and run native code",
        "  test <path>                    build and run native Severian tests",
        "  coverage <path>                report coverage support or source-map blocker",
        "  --emit <stage> <path>          emit hir, mir, mlir, stablehlo, llvm, or asm",
        "  clean [path]                   remove only the Severian project target directory",
        "  tree <path>                    print the Severian package dependency graph",
        "  metadata <path>                print Severian project metadata as JSON",
        "  explain <diagnostic-code>      explain a registered diagnostic",
        "  emit kinds: hir, mir, mlir, stablehlo, llvm, asm",
    ]
    .join("\n")
}
