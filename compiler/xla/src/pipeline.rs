use crate::{
    pjrt::{LoadedExecutable, PjrtClient},
    stablehlo::{export::PortableArtifactOptions, StableHloFormat, StableHloModule},
    Result, XlaError,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub optimization: OptimizationLevel,
    pub portable_artifact: bool,
    pub stablehlo_target_version: Option<String>,
    pub device_ordinal: Option<usize>,
    pub num_replicas: usize,
    pub num_partitions: usize,
    pub parameter_is_tupled_arguments: bool,
    pub use_spmd_partitioning: bool,
    pub debug_options: BTreeMap<String, String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimization: OptimizationLevel::O2,
            portable_artifact: false,
            stablehlo_target_version: None,
            device_ordinal: None,
            num_replicas: 1,
            num_partitions: 1,
            parameter_is_tupled_arguments: false,
            use_spmd_partitioning: false,
            debug_options: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct XlaPipeline {
    pub verify_stablehlo: bool,
    pub serialize_portable_artifact: bool,
}

impl Default for XlaPipeline {
    fn default() -> Self {
        Self {
            verify_stablehlo: true,
            serialize_portable_artifact: false,
        }
    }
}

impl XlaPipeline {
    pub fn prepare(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<StableHloModule> {
        if self.verify_stablehlo {
            module.validate_basic()?;
        }

        if !self.serialize_portable_artifact || !options.portable_artifact {
            return Ok(module.clone());
        }

        match module.format() {
            StableHloFormat::PortableArtifact => Ok(module.clone()),
            StableHloFormat::Text | StableHloFormat::MlirBytecode => {
                let artifact_options = PortableArtifactOptions {
                    target_version: options.stablehlo_target_version.clone(),
                    ..PortableArtifactOptions::default()
                };

                module.to_portable_artifact(&artifact_options)
            }
        }
    }

    pub fn compile(
        &self,
        client: &PjrtClient,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        let prepared = self.prepare(module, options)?;

        if options.num_replicas == 0 {
            return Err(XlaError::Compilation(
                "num_replicas must be greater than zero".into(),
            ));
        }
        if options.num_partitions == 0 {
            return Err(XlaError::Compilation(
                "num_partitions must be greater than zero".into(),
            ));
        }

        client.compile(&prepared, options)
    }
}
