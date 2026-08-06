use severian_driver::{
    compile_native, compile_native_tests, compile_path, compile_rocm, detect_amd_gpu_chip,
    lower_to_rocdl, native_test_compilation, run, run_integration_tests, run_tests,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Err(error) = execute(std::env::args().skip(1).collect()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn execute(args: Vec<String>) -> Result<(), String> {
    if args
        .first()
        .is_some_and(|argument| argument.ends_with(".sev"))
    {
        let source = PathBuf::from(&args[0]);
        if args.len() != 1 {
            return Err(usage());
        }
        return compile_and_run(&source);
    }
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage());
    };

    match command {
        "build" if args.len() <= 2 => {
            let directory = std::env::current_dir().map_err(|error| error.to_string())?;
            let manifests = if let Some(source) = args.get(1) {
                severian_package::find_manifest(Path::new(source))
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                severian_package::workspace_manifests(&directory)
                    .map_err(|error| error.to_string())?
            };
            let mut built_libraries = std::collections::HashSet::new();
            for manifest in manifests {
                let plan = severian_package::library_build_plan(&manifest)
                    .map_err(|error| error.to_string())?;
                for library in plan {
                    if !built_libraries.insert(library.artifact.clone()) {
                        continue;
                    }
                    compile_path(&library.source).map_err(|error| {
                        format!("could not build library `{}`: {error}", library.name)
                    })?;
                    severian_package::write_library_artifact(&library)
                        .map_err(|error| error.to_string())?;
                    println!("Built {} -> {}", library.name, library.artifact.display());
                }
            }
            let targets = if let Some(source) = args.get(1) {
                let source = PathBuf::from(source);
                let name = source
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("main")
                    .to_owned();
                vec![severian_package::BinaryTarget {
                    source,
                    name,
                    package_root: directory,
                }]
            } else {
                severian_package::workspace_binary_targets(&directory)
                    .map_err(|error| error.to_string())?
            };
            for target in targets {
                let output = target
                    .package_root
                    .join("target")
                    .join("debug")
                    .join(&target.name);
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                let compilation =
                    compile_path(&target.source).map_err(|error| error.to_string())?;
                if compilation.hir.main().is_none() && compilation.hir.test_count() == 0 {
                    continue;
                }
                compile_program_or_tests(&compilation, &output)?;
                println!("Built {} -> {}", target.name, output.display());
            }
        }
        "check" if args.len() == 2 => {
            compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
        }
        "emit-mlir" if args.len() == 2 => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            print!("{}", compilation.mlir);
        }
        "emit-mlir" if args.len() >= 4 && args[2] == "--target" && args[3] == "rocm" => {
            let chip = option_value(&args[4..], "--chip")
                .map(str::to_owned)
                .or_else(detect_amd_gpu_chip)
                .ok_or_else(|| {
                    "could not detect an AMD GPU; pass `--chip gfx…` or set SEVERIAN_AMDGPU_CHIP"
                        .to_string()
                })?;
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            let module = lower_to_rocdl(&compilation, &chip).map_err(|error| error.to_string())?;
            print!("{module}");
        }
        "emit-test-mlir" if args.len() == 2 => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            let (native, _) =
                native_test_compilation(&compilation).map_err(|error| error.to_string())?;
            print!("{}", native.mlir);
        }
        "compile"
            if (args.len() == 2 || args.len() == 4)
                && option_value(&args[2..], "--target") != Some("rocm") =>
        {
            let input = Path::new(&args[1]);
            let output = match args.as_slice() {
                [_, _, flag, output] if flag == "-o" => PathBuf::from(output),
                [_, _] => PathBuf::from("a.out"),
                _ => return Err(usage()),
            };
            let compilation = compile_path(input).map_err(|error| error.to_string())?;
            compile_native(&compilation, &output).map_err(|error| error.to_string())?;
            println!("{}", output.display());
        }
        "compile" if args.len() >= 4 && option_value(&args[2..], "--target") == Some("rocm") => {
            let input = Path::new(&args[1]);
            let output = option_value(&args[2..], "-o")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("a.out"));
            let chip = option_value(&args[2..], "--chip")
                .map(str::to_owned)
                .or_else(detect_amd_gpu_chip)
                .ok_or_else(|| {
                    "could not detect an AMD GPU; pass `--chip gfx…` or set SEVERIAN_AMDGPU_CHIP"
                        .to_string()
                })?;
            let compilation = compile_path(input).map_err(|error| error.to_string())?;
            compile_rocm(&compilation, &output, &chip).map_err(|error| error.to_string())?;
            println!("{} ({chip})", output.display());
        }
        "compile-tests" if args.len() == 4 && args[2] == "-o" => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            let output = PathBuf::from(&args[3]);
            let count =
                compile_native_tests(&compilation, &output).map_err(|error| error.to_string())?;
            println!("{} ({count} native tests)", output.display());
        }
        "run" if args.len() == 2 => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            run(&compilation.hir, |line| println!("{line}")).map_err(|error| error.to_string())?;
        }
        "test" if args.len() == 2 || args.len() == 3 => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            let mode = args.get(2).map(String::as_str);
            if !matches!(
                mode,
                None | Some("--integration") | Some("--integration-only")
            ) {
                return Err(usage());
            }
            let mut passed = 0;
            if mode != Some("--integration-only") {
                passed += run_tests(&compilation.hir, |line| println!("{line}"))
                    .map_err(|error| error.to_string())?;
            }
            if matches!(mode, Some("--integration") | Some("--integration-only")) {
                passed += run_integration_tests(&compilation, |line| println!("{line}"))
                    .map_err(|error| error.to_string())?;
            }
            println!("{passed} passed");
        }
        "help" | "--help" | "-h" => println!("{}", usage()),
        _ => return Err(usage()),
    }

    Ok(())
}

fn usage() -> String {
    concat!(
        "usage: sev [command] [source.sev] [options]\n",
        "  sev source.sev: compile to a temporary native executable and run it\n",
        "  build [source.sev]: compile into target/debug, using Severian.toml by default\n",
        "  check source.sev: run the frontend and ownership checks\n",
        "  emit-mlir target options: --target rocm [--chip gfx1100]\n",
        "  compile options: [-o executable] [--target rocm [--chip gfx1101]]\n",
        "  compile-tests options: -o executable\n",
        "  run source.sev: execute through the controlled development runtime\n",
        "  test options: --integration | --integration-only"
    )
    .into()
}

fn compile_program_or_tests(
    compilation: &severian_driver::Compilation,
    output: &Path,
) -> Result<(), String> {
    if compilation.hir.main().is_some() {
        compile_native(compilation, output).map_err(|error| error.to_string())
    } else if compilation.hir.test_count() > 0 {
        compile_native_tests(compilation, output)
            .map(|_| ())
            .map_err(|error| error.to_string())
    } else {
        Err("source has neither `main()` nor native tests".into())
    }
}

fn compile_and_run(source: &Path) -> Result<(), String> {
    let compilation = compile_path(source).map_err(|error| error.to_string())?;
    let executable = std::env::temp_dir().join(format!(
        "severian-run-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos()
    ));
    compile_program_or_tests(&compilation, &executable)?;
    let status = Command::new(&executable)
        .status()
        .map_err(|error| format!("could not run {}: {error}", executable.display()));
    let _ = std::fs::remove_file(&executable);
    let status = status?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("native executable exited with {status}"))
    }
}

fn option_value<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].as_str())
}
