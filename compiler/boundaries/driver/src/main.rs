use severian_driver::compile_file;
use std::path::PathBuf;

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let command = arguments.next().ok_or_else(usage)?;
    if command != "compile" {
        return Err(usage());
    }
    let source = PathBuf::from(arguments.next().ok_or_else(usage)?);
    let flag = arguments.next().ok_or_else(usage)?;
    if flag != "-o" {
        return Err(usage());
    }
    let output = PathBuf::from(arguments.next().ok_or_else(usage)?);
    if arguments.next().is_some() {
        return Err(usage());
    }
    compile_file(&source, &output).map_err(|error| error.to_string())?;
    Ok(())
}

fn usage() -> String {
    "usage: sev compile <input.sev> -o <executable>".into()
}
