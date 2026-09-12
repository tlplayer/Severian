//! Native CLI profiling. Summary accounting stays in the compiler process.
mod analysis;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Default)]
pub struct Options {
    pub mode: Option<String>,
    pub output: Option<PathBuf>,
    pub arguments: Vec<String>,
}

pub const HELP: &str = "\nprofiling:\n  --profile [calls|memory|time]  Hook calls, allocations, and elapsed time.\n  --profile-output DIR        New directory for the native report and traces.\n  --build-profile NAME        Build settings such as release or dev.\n";

pub fn parse(arguments: Vec<String>) -> Result<Options, String> {
    let mut result = Options::default();
    let mut cursor = 0;
    while cursor < arguments.len() {
        let argument = &arguments[cursor];
        if argument == "--" {
            result.arguments.extend_from_slice(&arguments[cursor..]);
            break;
        }
        let (name, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(n, v)| (n, Some(v)));
        if name == "--profile" {
            if result.mode.is_some() {
                return Err("--profile may only be supplied once".into());
            }
            let mut mode = inline.unwrap_or("time");
            if inline.is_none() {
                if let Some(next) = arguments.get(cursor + 1) {
                    if matches!(next.as_str(), "calls" | "cpu" | "memory" | "time" | "all" | "summary") {
                        mode = next;
                        cursor += 1;
                    } else if matches!(next.as_str(), "release" | "dev" | "debug") {
                        return Err(format!("use --build-profile {next} for build settings"));
                    }
                }
            }
            if !matches!(mode, "calls" | "cpu" | "memory" | "time" | "all" | "summary") {
                return Err("--profile expects calls, memory, or time".into());
            }
            result.mode = Some(
                if mode == "summary" { "time" }
                else if matches!(mode, "all" | "cpu") { "calls" }
                else {
                    mode
                }
                .into(),
            );
        } else if name == "--profile-output" {
            let value = if let Some(value) = inline {
                value
            } else {
                cursor += 1;
                arguments
                    .get(cursor)
                    .filter(|v| v.as_str() != "--")
                    .ok_or("--profile-output requires a directory")?
            };
            if value.is_empty() {
                return Err("--profile-output requires a directory".into());
            }
            result.output = Some(PathBuf::from(value));
        } else {
            result.arguments.push(argument.clone());
            if inline.is_none()
                && matches!(
                    name,
                    "--build-profile"
                        | "--target"
                        | "--bin"
                        | "--emit"
                        | "--sysroot"
                        | "--filter"
                        | "--alias"
                        | "-o"
                        | "--output"
                )
            {
                cursor += 1;
                if let Some(value) = arguments.get(cursor) {
                    result.arguments.push(value.clone());
                }
            }
        }
        cursor += 1;
    }
    if result.output.is_some() && result.mode.is_none() {
        return Err("--profile-output requires --profile".into());
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct Usage {
    user: f64,
    system: f64,
    peak: i64,
}

fn usage(who: libc::c_int) -> Result<Usage, String> {
    let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // getrusage initializes the supplied structure on success.
    if unsafe { libc::getrusage(who, value.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let value = unsafe { value.assume_init() };
    let seconds = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 / 1_000_000.0;
    let peak = value.ru_maxrss as i64;
    #[cfg(target_os = "macos")]
    let peak = peak / 1024;
    Ok(Usage {
        user: seconds(value.ru_utime),
        system: seconds(value.ru_stime),
        peak,
    })
}

pub struct Session {
    start: Instant,
    own: Usage,
    children: Usage,
    pub directory: PathBuf,
}

impl Session {
    pub fn begin(options: &Options) -> Result<Self, String> {
        let directory = options.output.clone().unwrap_or_else(|| {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            PathBuf::from(format!(
                "package.pkg/debug/profiles/rust-{}-{nonce}",
                std::process::id()
            ))
        });
        if let Some(parent) = directory.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::create_dir(&directory).map_err(|e| format!("cannot create profile directory: {e}"))?;
        let directory = fs::canonicalize(directory).map_err(|e| e.to_string())?;
        env::set_var("SEVERIAN_PROFILE_ACTIVE", "1");
        if matches!(options.mode.as_deref(), Some("calls" | "memory")) {
            fs::write(directory.join("hooks.jsonl"), "").map_err(|error| error.to_string())?;
            env::set_var("SEVERIAN_HOOK_RECORDS", directory.join("hooks.jsonl"));
        }
        Ok(Self {
            start: Instant::now(),
            own: usage(libc::RUSAGE_SELF)?,
            children: usage(libc::RUSAGE_CHILDREN)?,
            directory,
        })
    }

    pub fn finish(&self, options: &Options, code: i32) -> Result<(), String> {
        let wall = self.start.elapsed().as_secs_f64();
        let own = usage(libc::RUSAGE_SELF)?;
        let children = usage(libc::RUSAGE_CHILDREN)?;
        let user = own.user - self.own.user + children.user - self.children.user;
        let system = own.system - self.own.system + children.system - self.children.system;
        let peak = own.peak.max(children.peak);
        eprintln!("Profile breakdown:\n  Time: {wall:.3}s wall\n  CPU: {:.3}s total ({user:.3}s user, {system:.3}s system)\n  Memory: {:.2} MiB maximum process RSS\n  Exit: {code}", user + system, peak as f64 / 1024.0);
        let report = serde_json::json!({"format": 1, "compiler": "rust", "mode": options.mode,
            "binary": env::current_exe().map_err(|e| e.to_string())?, "arguments": options.arguments,
            "cwd": env::current_dir().map_err(|e| e.to_string())?, "exit_code": code,
            "wall_seconds": wall, "user_seconds": user, "system_seconds": system,
            "peak_rss_kib": peak});
        fs::write(
            self.directory.join("report.json"),
            report.to_string() + "\n",
        )
        .map_err(|e| e.to_string())?;
        eprintln!("Profile: {}", self.directory.join("report.json").display());
        if matches!(options.mode.as_deref(), Some("calls" | "memory")) {
            severian_driver::hooks::report(&self.directory.join("hooks.jsonl"), &self.directory.join("functions.tsv"))?;
            eprintln!("Functions: {}", self.directory.join("functions.tsv").display());
        }
        Ok(())
    }

    pub fn capture(&self, options: &Options) -> Result<i32, String> {
        let binary = env::current_exe().map_err(|e| e.to_string())?;
        let mut command = Command::new("heaptrack");
        command.args(["--record-only", "-o"]).arg(self.directory.join("heaptrack"));
        let result = command.arg(binary).args(&options.arguments).status();
        let code = match result {
            Ok(status) => {
                use std::os::unix::process::ExitStatusExt;
                status
                    .code()
                    .unwrap_or_else(|| 128 + status.signal().unwrap_or(1))
            }
            Err(error) => {
                self.finish(options, 127)?;
                eprintln!(
                    "{} stack capture could not start: {error}",
                    options.mode.as_deref().unwrap_or("")
                );
                return Ok(127);
            }
        };
        self.finish(options, code)?;
        let mut analyses = Vec::new();
        {
            for entry in fs::read_dir(&self.directory).map_err(|e| e.to_string())? {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("heaptrack"))
                    && matches!(
                        path.extension().and_then(|s| s.to_str()),
                        Some("gz" | "zst")
                    )
                {
                    let mut analysis = Command::new("heaptrack_print");
                    let name = format!("memory-{}.txt", analyses.len());
                    analysis
                        .arg("-f")
                        .arg(&path)
                        .args([
                            "--merge-backtraces",
                            "0",
                            "--print-leaks",
                            "1",
                            "--peak-limit",
                            "25",
                            "--print-flamegraph",
                        ])
                        .arg(self.directory.join(format!("{name}.allocations.folded")));
                    analyses.push((name, analysis));
                }
            }
        }
        let mut analysis_code = if analyses.is_empty() { 1 } else { 0 };
        for (name, mut analysis) in analyses {
            let result = match analysis.output() {
                Ok(result) => result,
                Err(error) => {
                    analysis_code = 127;
                    fs::write(
                        self.directory.join(format!("{name}.stderr")),
                        error.to_string(),
                    )
                    .map_err(|e| e.to_string())?;
                    continue;
                }
            };
            fs::write(self.directory.join(&name), &result.stdout).map_err(|e| e.to_string())?;
            fs::write(self.directory.join(format!("{name}.stderr")), result.stderr)
                .map_err(|e| e.to_string())?;
            if !result.status.success() {
                analysis_code = result.status.code().unwrap_or(1);
            } else {
                let trace = PathBuf::from(
                    analysis
                        .get_args()
                        .nth(1)
                        .ok_or("missing allocation trace")?,
                );
                if let Err(error) = analysis::memory(
                    &self.directory,
                    &trace,
                    &name,
                    &String::from_utf8_lossy(&result.stdout),
                ) {
                    eprintln!("Function analysis: {error}");
                    analysis_code = 1;
                }
            }
            eprintln!("Stacks: {}", self.directory.join(name).display());
        }
        fs::write(
            self.directory.join("analysis.status"),
            format!("{analysis_code}\n"),
        )
        .map_err(|e| e.to_string())?;
        Ok(if code != 0 { code } else { analysis_code })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).into()).collect()
    }
    #[test]
    fn flags_preserve_file_names_values_and_program_arguments() {
        let options = parse(strings(&[
            "run",
            "--profile",
            "file.sev",
            "-o",
            "--profile",
            "--",
            "--profile",
            "cpu",
        ]))
        .unwrap();
        assert_eq!(options.mode.as_deref(), Some("time"));
        assert_eq!(
            options.arguments,
            strings(&[
                "run",
                "file.sev",
                "-o",
                "--profile",
                "--",
                "--profile",
                "cpu"
            ])
        );
        for command in ["build", "test", "run"] {
            for mode in ["calls", "cpu", "memory", "time"] {
                assert_eq!(
                    parse(strings(&[command, "--profile", mode]))
                        .unwrap()
                        .mode
                        .as_deref(),
                    Some(if mode == "cpu" { "calls" } else { mode })
                );
            }
        }
    }
    #[test]
    fn build_settings_and_invalid_modes_are_unambiguous() {
        assert_eq!(
            parse(strings(&["--build-profile", "release", "--profile=cpu"]))
                .unwrap()
                .arguments,
            strings(&["--build-profile", "release"])
        );
        assert!(parse(strings(&["--profile", "release"])).is_err());
        assert!(parse(strings(&["--profile=wat"])).is_err());
        assert!(parse(strings(&["--profile", "--profile"])).is_err());
    }
    #[test]
    fn native_accounting_returns_cpu_and_resident_memory() {
        let value = usage(libc::RUSAGE_SELF).unwrap();
        assert!(value.user >= 0.0 && value.system >= 0.0 && value.peak > 0);
    }
}
