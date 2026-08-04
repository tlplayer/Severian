use severian_driver::{
    compile_native, compile_native_tests, compile_path, compile_rocm, detect_amd_gpu_chip,
    lower_to_rocdl, native_test_compilation, run, run_integration_tests, run_tests,
};
use std::path::{Path, PathBuf};

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

    match command {
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
        "usage: sev <check|emit-mlir|emit-test-mlir|compile|compile-tests|run|test> <source.sev> [options]\n",
        "  emit-mlir target options: --target rocm [--chip gfx1100]\n",
        "  compile options: [-o executable] [--target rocm [--chip gfx1101]]\n",
        "  compile-tests options: -o executable\n",
        "  test options: --integration | --integration-only"
    )
    .into()
}

fn option_value<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].as_str())
}
