//! Compiler update bootstrap. Build directly with rustc; no Python or Cargo deps.
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Default)]
struct Options {
    repository: PathBuf,
    local: bool,
    no_install: bool,
    install_dir: Option<PathBuf>,
    max_errors: Option<u32>,
    help: bool,
}

fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Options> {
    let mut options = Options::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--repository" => options.repository = arguments.next().ok_or("missing repository")?.into(),
            "--local" => options.local = true,
            "--no-install" => options.no_install = true,
            "--install-dir" => options.install_dir = Some(arguments.next().ok_or("missing install directory")?.into()),
            "--max-errors" => {
                let value = arguments.next().ok_or("missing error limit")?.parse::<u32>()?;
                if value > 999_999_999 { return Err("--max-errors must be between 0 and 999999999".into()); }
                options.max_errors = Some(value);
            }
            "-h" | "--help" => options.help = true,
            _ => return Err(format!("unknown compiler update option: {argument}").into()),
        }
    }
    Ok(options)
}

fn git(root: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git").args(arguments).current_dir(root).output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

fn record(reports: &Path, stage: &str, category: &str, message: &str) -> Result<()> {
    let directory = reports.join(stage).join(category);
    fs::create_dir_all(&directory)?;
    // These records are also consumed by the compiler's grouped report writer.
    fs::write(directory.join("update.json"), format!(
        "{{\"source\":{},\"message\":{}}}\n",
        json_string(&format!("build/{stage}")), json_string(message)))?;
    let mut log = OpenOptions::new().create(true).append(true).open(reports.join("log.txt"))?;
    writeln!(log, "\n===== build/{stage} =====\n[{stage}/{category}]\n{message}")?;
    Ok(())
}

fn run(reports: &Path, stage: &str, command: &mut Command) -> Result<()> {
    let transcript = reports.join(format!("{stage}.txt"));
    let output = File::create(&transcript)?;
    let description = format!("command: {command:?}");
    let status = command.stdout(Stdio::from(output.try_clone()?)).stderr(Stdio::from(output)).status();
    let text = fs::read_to_string(&transcript)?;
    record(reports, stage, "trace", &format!("{description}\n{text}"))?;
    match status {
        Ok(status) if status.success() => Ok(()),
        other => {
            let message = format!("{description}\n{text}\nresult: {other:?}");
            record(reports, stage, "error", &message)?;
            Err(format!("{stage} failed; see {}", reports.join("log.txt").display()).into())
        }
    }
}

fn install(root: &Path, directory: &Path) -> Result<()> {
    fs::create_dir_all(directory)?;
    for name in ["sev", "sev_rust"] {
        let destination = directory.join(name);
        let source = root.join("bin").join(name);
        if fs::read_link(&destination).ok().as_ref() == Some(&source) { continue; }
        if fs::symlink_metadata(&destination).is_ok() {
            let backup = directory.join(format!("{name}.before-source-default"));
            if fs::symlink_metadata(&backup).is_err() {
                if let Ok(target) = fs::read_link(&destination) {
                    std::os::unix::fs::symlink(target, &backup)?;
                } else {
                    fs::copy(&destination, &backup)?;
                }
            }
        }
        let temporary = directory.join(format!(".{name}.install-{}", std::process::id()));
        std::os::unix::fs::symlink(&source, &temporary)?;
        fs::rename(temporary, destination)?;
    }
    Ok(())
}

fn update(options: Options) -> Result<()> {
    if options.help {
        println!("sev update [--local] [--no-install] [--install-dir PATH] [--max-errors N]\nBuild the Rust bootstrap and source compiler; install after verification.");
        return Ok(());
    }
    let root = options.repository.canonicalize()?;
    let cache = root.join("package.pkg");
    fs::create_dir_all(&cache)?;
    let lock = OpenOptions::new().create(true).truncate(false).write(true).open(cache.join("compiler-update.lock"))?;
    lock.try_lock().map_err(|_| "another compiler update is running")?;
    let reports = cache.join("debug/build");
    if reports.exists() {
        let previous = cache.join(format!("debug/build-before-update-{}", std::process::id()));
        fs::rename(&reports, previous)?;
    }
    fs::create_dir_all(&reports)?;
    fs::write(reports.join("log.txt"), "Build trace (grouped by source file)\n")?;
    println!("Build log: {}", reports.join("log.txt").display());
    env::set_var("SEVERIAN_BUILD_REPORT_DIR", &reports);
    env::set_var("SEVERIAN_SYSROOT", &root);
    if let Some(limit) = options.max_errors { env::set_var("SEVERIAN_MAX_ERRORS", limit.to_string()); }
    if !options.local {
        if !git(&root, &["status", "--porcelain", "--untracked-files=normal"])?.is_empty() {
            return Err("checkout has local changes; use sev update --local or commit before updating".into());
        }
        let branch = git(&root, &["symbolic-ref", "--short", "HEAD"])?;
        let remote = git(&root, &["config", &format!("branch.{branch}.remote")])?;
        if remote == "." { return Err("current branch has no remote upstream".into()); }
        let upstream = git(&root, &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])?;
        git(&root, &["fetch", &remote])?;
        println!("{}", git(&root, &["merge", "--ff-only", &upstream])?);
    }
    println!("Building Rust bootstrap (severian-driver)");
    if let Err(error) = run(&reports, "rust-bootstrap", Command::new("cargo").current_dir(&root)
        .args(["build", "--release", "--keep-going", "--target-dir"]).arg(&cache)
        .args(["-p", "severian-driver", "--bin", "sev"])) {
        record(&reports, "source-compiler", "error", "blocked: Rust bootstrap build failed; source compiler was not built")?;
        return Err(error);
    }
    let seed = cache.join("release/sev");
    let candidate_directory = cache.join("compiler-update");
    fs::create_dir_all(&candidate_directory)?;
    let candidate = candidate_directory.join("sev_compiler");
    println!("Building source compiler (sev_compiler) and its packages");
    run(&reports, "source-compiler", Command::new(&seed).current_dir(&root)
        .arg("build").arg(root.join("sev_compiler"))
        .args(["--bin", "sev_compiler", "--build-profile", "release", "-o"]).arg(&candidate))?;
    run(&reports, "rust-version", Command::new(&seed).arg("--version"))?;
    run(&reports, "source-help", Command::new(&candidate).arg("--help"))?;
    let smoke = candidate_directory.join("smoke.sev");
    fs::write(&smoke, "assert(20 + 22 == 42)\n")?;
    // timeout manages the child's process group, including native subprocesses.
    run(&reports, "source-smoke", Command::new("timeout").current_dir(&root)
        .args(["--kill-after=5", "90"]).arg(&candidate).arg(&smoke).arg("--sysroot").arg(&root))?;
    let destination = cache.join("bin/sev_compiler");
    fs::create_dir_all(destination.parent().ok_or("missing compiler directory")?)?;
    if destination.is_file() {
        fs::copy(&destination, candidate_directory.join("previous-sev_compiler"))?;
    }
    // Promote only a verified candidate. Failures leave the installed binary intact.
    fs::rename(candidate, &destination)?;
    if !options.no_install {
        let directory = options.install_dir.or_else(|| env::var_os("CARGO_HOME").map(|p| PathBuf::from(p).join("bin")))
            .or_else(|| env::var_os("HOME").map(|p| PathBuf::from(p).join(".cargo/bin")))
            .ok_or("set --install-dir or CARGO_HOME")?;
        install(&root, &directory)?;
    }
    println!("Compiler updated ({}): sev = source compiler; sev_rust = Rust backup.", git(&root, &["rev-parse", "--short", "HEAD"])?);
    Ok(())
}

fn main() {
    if let Err(error) = parse(env::args().skip(1)).and_then(update) {
        eprintln!("Compiler update failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_options_reject_invalid_limits_and_missing_values() {
        assert!(parse(["--max-errors".into(), "-1".into()]).is_err());
        assert!(parse(["--max-errors".into(), "1000000000".into()]).is_err());
        assert!(parse(["--install-dir".into()]).is_err());
        let options = parse(["--local".into(), "--no-install".into(), "--max-errors".into(), "0".into()]).unwrap();
        assert!(options.local && options.no_install);
        assert_eq!(options.max_errors, Some(0));
    }

    #[test]
    fn report_strings_escape_paths_and_control_characters() {
        assert_eq!(json_string("a\"b\\c\n\t\r\0"), "\"a\\\"b\\\\c\\n\\t\\r\\u0000\"");
    }
}
