use serde::Deserialize;
use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
pub struct CoverageToolchain {
    pub llvm_profdata: PathBuf,
    pub llvm_cov: PathBuf,
}

impl CoverageToolchain {
    pub fn discover() -> io::Result<Self> {
        Ok(Self {
            llvm_profdata: find_tool(&["llvm-profdata", "llvm-profdata-21"])?,
            llvm_cov: find_tool(&["llvm-cov", "llvm-cov-21"])?,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoverageMetric {
    pub count: u64,
    pub covered: u64,
    pub percent: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    pub lines: CoverageMetric,
    pub regions: CoverageMetric,
    pub branches: CoverageMetric,
    pub functions: CoverageMetric,
    pub raw_json: Option<PathBuf>,
}

pub fn merge_profiles(
    toolchain: &CoverageToolchain,
    raw_profiles: &[PathBuf],
    output: &Path,
) -> io::Result<()> {
    if raw_profiles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no .profraw files were provided",
        ));
    }

    let mut arguments = vec![OsString::from("merge"), OsString::from("-sparse")];
    arguments.extend(
        raw_profiles
            .iter()
            .map(|profile| profile.as_os_str().to_owned()),
    );
    arguments.push("-o".into());
    arguments.push(output.as_os_str().to_owned());

    run(&toolchain.llvm_profdata, &arguments)
}

pub fn discover_raw_profiles(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut profiles = Vec::new();
    if !directory.exists() {
        return Ok(profiles);
    }

    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("profraw") {
            profiles.push(path);
        }
    }
    profiles.sort();
    Ok(profiles)
}

pub fn export_report(
    toolchain: &CoverageToolchain,
    binary: &Path,
    objects: &[PathBuf],
    profile: &Path,
    output_json: &Path,
) -> io::Result<CoverageReport> {
    let mut command = Command::new(&toolchain.llvm_cov);
    command
        .arg("export")
        .arg(binary)
        .arg(format!("-instr-profile={}", profile.display()));

    for object in objects {
        command.arg("-object").arg(object);
    }

    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{} export failed: {}",
            toolchain.llvm_cov.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    fs::write(output_json, &output.stdout)?;
    let export: LlvmCovExport = serde_json::from_slice(&output.stdout)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    let totals = export
        .data
        .first()
        .map(|data| &data.totals)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "llvm-cov export had no data"))?;

    Ok(CoverageReport {
        lines: metric(&totals.lines),
        regions: metric(&totals.regions),
        branches: metric(&totals.branches),
        functions: metric(&totals.functions),
        raw_json: Some(output_json.to_owned()),
    })
}

pub fn render_text(report: &CoverageReport) -> String {
    format!(
        "Coverage\n\nLines      {:6.2}% ({}/{})\nRegions    {:6.2}% ({}/{})\nBranches   {:6.2}% ({}/{})\nFunctions  {:6.2}% ({}/{})\n",
        report.lines.percent,
        report.lines.covered,
        report.lines.count,
        report.regions.percent,
        report.regions.covered,
        report.regions.count,
        report.branches.percent,
        report.branches.covered,
        report.branches.count,
        report.functions.percent,
        report.functions.covered,
        report.functions.count,
    )
}

#[derive(Debug, Deserialize)]
struct LlvmCovExport {
    #[serde(default)]
    data: Vec<LlvmCovData>,
}

#[derive(Debug, Deserialize)]
struct LlvmCovData {
    totals: LlvmCovTotals,
}

#[derive(Debug, Deserialize)]
struct LlvmCovTotals {
    lines: LlvmCovMetric,
    regions: LlvmCovMetric,
    branches: LlvmCovMetric,
    functions: LlvmCovMetric,
}

#[derive(Debug, Deserialize)]
struct LlvmCovMetric {
    count: u64,
    covered: u64,
    percent: f64,
}

fn metric(metric: &LlvmCovMetric) -> CoverageMetric {
    CoverageMetric {
        count: metric.count,
        covered: metric.covered,
        percent: metric.percent,
    }
}

fn find_tool(candidates: &[&str]) -> io::Result<PathBuf> {
    for candidate in candidates {
        if let Some(path) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&path) {
                let executable = directory.join(candidate);
                if executable.is_file() {
                    return Ok(executable);
                }
            }
        }

        let direct = Path::new(candidate);
        if direct.components().count() > 1 && direct.is_file() {
            return Ok(direct.to_owned());
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("could not find any of: {}", candidates.join(", ")),
    ))
}

fn run(tool: &Path, arguments: &[OsString]) -> io::Result<()> {
    let output = Command::new(tool).args(arguments).output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{} failed: {}",
            tool.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}
