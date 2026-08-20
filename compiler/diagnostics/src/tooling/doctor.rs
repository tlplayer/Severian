use crate::{Diagnostic, DiagnosticBag};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
pub struct ToolRequirement {
    pub name: &'static str,
    pub candidates: &'static [&'static str],
    pub required_for: &'static str,
    pub required: bool,
}

pub fn standard_requirements() -> Vec<ToolRequirement> {
    vec![
        ToolRequirement {
            name: "mlir-opt",
            candidates: &["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"],
            required_for: "MLIR lowering and verification",
            required: true,
        },
        ToolRequirement {
            name: "mlir-translate",
            candidates: &[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ],
            required_for: "LLVM IR translation",
            required: true,
        },
        ToolRequirement {
            name: "clang",
            candidates: &["clang", "clang-21"],
            required_for: "native linking",
            required: true,
        },
        ToolRequirement {
            name: "llvm-profdata",
            candidates: &["llvm-profdata", "llvm-profdata-21"],
            required_for: "coverage profile merging",
            required: false,
        },
        ToolRequirement {
            name: "llvm-cov",
            candidates: &["llvm-cov", "llvm-cov-21"],
            required_for: "coverage reports",
            required: false,
        },
        ToolRequirement {
            name: "stablehlo-translate",
            candidates: &["stablehlo-translate"],
            required_for: "StableHLO portable artifacts",
            required: false,
        },
        ToolRequirement {
            name: "nvidia-smi",
            candidates: &["nvidia-smi"],
            required_for: "NVIDIA device discovery",
            required: false,
        },
        ToolRequirement {
            name: "rocminfo",
            candidates: &["rocminfo", "/opt/rocm/bin/rocminfo"],
            required_for: "AMD device discovery",
            required: false,
        },
    ]
}

pub fn run() -> DiagnosticBag {
    let mut bag = DiagnosticBag::default();

    for requirement in standard_requirements() {
        match find_executable(requirement.candidates) {
            Some(path) => {
                if version(&path).is_none() {
                    bag.push(Diagnostic::warning(
                        "doctor::tool-version",
                        format!(
                            "{} was found at {} but its version could not be queried",
                            requirement.name,
                            path.display()
                        ),
                    ));
                }
            }
            None if requirement.required => {
                bag.push(
                    Diagnostic::error(
                        "doctor::missing-tool",
                        format!(
                            "required tool `{}` was not found ({})",
                            requirement.name, requirement.required_for
                        ),
                    )
                    .with_help("install the matching LLVM/MLIR toolchain or configure its path"),
                );
            }
            None => {
                bag.push(Diagnostic::warning(
                    "doctor::optional-tool",
                    format!(
                        "optional tool `{}` was not found; {} will be unavailable",
                        requirement.name, requirement.required_for
                    ),
                ));
            }
        }
    }

    bag
}

fn find_executable(candidates: &[&str]) -> Option<PathBuf> {
    for candidate in candidates {
        let candidate_path = Path::new(candidate);
        if candidate_path.components().count() > 1 && candidate_path.is_file() {
            return Some(candidate_path.to_owned());
        }

        let Some(path) = std::env::var_os("PATH") else {
            continue;
        };
        for directory in std::env::split_paths(&path) {
            let executable = directory.join(candidate);
            if executable.is_file() {
                return Some(executable);
            }
        }
    }
    None
}

fn version(tool: &Path) -> Option<String> {
    let output = Command::new(tool).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let bytes = if output.stdout.is_empty() {
        &output.stderr
    } else {
        &output.stdout
    };
    Some(
        String::from_utf8_lossy(bytes)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned(),
    )
}
