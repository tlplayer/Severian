use severian_driver::{compile_native, compile_path, run};
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
        "compile" if args.len() == 2 || args.len() == 4 => {
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
        "run" if args.len() == 2 => {
            let compilation =
                compile_path(Path::new(&args[1])).map_err(|error| error.to_string())?;
            run(&compilation.hir, |line| println!("{line}")).map_err(|error| error.to_string())?;
        }
        "help" | "--help" | "-h" => println!("{}", usage()),
        _ => return Err(usage()),
    }

    Ok(())
}

fn usage() -> String {
    "usage: sev <check|emit-mlir|compile|run> <source.sev> [ -o executable ]".into()
}
