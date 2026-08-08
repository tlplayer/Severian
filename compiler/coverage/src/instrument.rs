use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageMode {
    Line,
    Branch,
    Full,
}

#[derive(Debug, Clone)]
pub struct CoverageInstrumentationPlan {
    pub mode: CoverageMode,
    pub source_map: PathBuf,
    pub profile_directory: PathBuf,
    pub profile_pattern: String,
    pub preserve_names: bool,
}

impl CoverageInstrumentationPlan {
    pub fn new(
        target_directory: impl AsRef<Path>,
        binary_name: &str,
    ) -> Self {
        let coverage_directory = target_directory.as_ref().join("coverage");
        Self {
            mode: CoverageMode::Full,
            source_map: coverage_directory.join(format!("{binary_name}.coverage-map.json")),
            profile_directory: coverage_directory.join("profiles"),
            profile_pattern: format!("{binary_name}-%m-%p.profraw"),
            preserve_names: true,
        }
    }

    pub fn environment(&self) -> CoverageProfileEnvironment {
        CoverageProfileEnvironment {
            values: BTreeMap::from([(
                "LLVM_PROFILE_FILE".into(),
                self.profile_directory
                    .join(&self.profile_pattern)
                    .to_string_lossy()
                    .into_owned(),
            )]),
        }
    }

    pub fn prepare_directories(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.profile_directory)?;
        if let Some(parent) = self.source_map.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }

    /// Required lowering behavior for source-based LLVM coverage.
    ///
    /// These are semantic requirements, not command-line guesses. The lowering
    /// layer must emit counter intrinsics plus LLVM coverage mapping records
    /// tied to Severian source locations. Merely adding profiling counters is
    /// insufficient for source-based `llvm-cov` reports.
    pub fn lowering_requirements(&self) -> &'static [&'static str] {
        &[
            "preserve Severian source spans through HIR and MLIR locations",
            "assign stable coverage region ids",
            "emit LLVM instrprof counters for instrumented regions",
            "emit LLVM coverage mapping records referencing Severian files",
            "retain coverage globals/functions through optimization and linking",
            "link the LLVM profile runtime",
        ]
    }

    /// Suggested optimizer policy while coverage is enabled.
    pub fn optimization_policy(&self) -> CoverageOptimizationPolicy {
        CoverageOptimizationPolicy {
            optimization_level: 0,
            disable_dead_code_elimination_of_covered_functions: true,
            preserve_debug_locations: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoverageProfileEnvironment {
    pub values: BTreeMap<String, String>,
}

impl CoverageProfileEnvironment {
    pub fn apply(&self, command: &mut std::process::Command) {
        command.envs(&self.values);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoverageOptimizationPolicy {
    pub optimization_level: u8,
    pub disable_dead_code_elimination_of_covered_functions: bool,
    pub preserve_debug_locations: bool,
}
