use super::{
    debug::DebugOptions,
    device_assignment::DeviceAssignment,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationEffort {
    /// Preserve XLA's default/unknown effort selection.
    Default,
    /// Deprecated float effort retained for compatibility with XLA.
    Relative(f32),
}

impl OptimizationEffort {
    pub fn relative(self) -> Option<f32> {
        match self {
            Self::Default => None,
            Self::Relative(value) => Some(value.clamp(-1.0, 1.0)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortLevel {
    Unknown,
    O0,
    O1,
    O2,
    O3,
}

impl EffortLevel {
    pub const fn protobuf_value(self) -> i64 {
        match self {
            Self::Unknown => 0,
            Self::O0 => 9,
            Self::O1 => 19,
            Self::O2 => 29,
            Self::O3 => 39,
        }
    }
}

#[derive(Debug, Clone)]
pub struct XlaCompileOptions {
    pub parameter_is_tupled_arguments: bool,
    pub compile_portable_executable: bool,
    pub profile_version: i64,
    pub serialized_multi_slice_config: Vec<u8>,
    pub allow_in_place_mlir_modification: bool,
    pub compiler_variant: Option<String>,

    pub device_ordinal: Option<i64>,
    pub num_replicas: i64,
    pub num_partitions: i64,
    pub use_spmd_partitioning: bool,
    pub use_auto_spmd_partitioning: bool,
    pub use_shardy_partitioner: bool,

    pub execution_time_effort: OptimizationEffort,
    pub memory_fitting_effort: OptimizationEffort,
    pub optimization_level: EffortLevel,
    pub memory_fitting_level: EffortLevel,

    pub deduplicate_hlo: bool,
    pub alias_passthrough_params: bool,
    pub run_backend_only: bool,
    pub device_memory_size: Option<i64>,

    pub device_assignment: Option<DeviceAssignment>,

    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,

    pub auto_spmd_partitioning_mesh_shape: Vec<i64>,
    pub auto_spmd_partitioning_mesh_ids: Vec<i64>,

    pub process_index: i64,
    pub process_count: i64,

    /// Serialized FDO/profile blob consumed by the backend.
    pub fdo_profile: Vec<u8>,

    /// CompileOptionsProto.env_option_overrides.
    pub debug_options: DebugOptions,
}

impl Default for XlaCompileOptions {
    fn default() -> Self {
        Self {
            parameter_is_tupled_arguments: false,
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_config: Vec::new(),
            allow_in_place_mlir_modification: false,
            compiler_variant: None,

            device_ordinal: None,
            num_replicas: 1,
            num_partitions: 1,
            use_spmd_partitioning: false,
            use_auto_spmd_partitioning: false,
            use_shardy_partitioner: false,

            execution_time_effort: OptimizationEffort::Default,
            memory_fitting_effort: OptimizationEffort::Default,
            optimization_level: EffortLevel::Unknown,
            memory_fitting_level: EffortLevel::Unknown,

            deduplicate_hlo: false,
            alias_passthrough_params: false,
            run_backend_only: false,
            device_memory_size: None,

            device_assignment: None,

            allow_spmd_sharding_propagation_to_parameters: vec![false],
            allow_spmd_sharding_propagation_to_output: vec![false],

            auto_spmd_partitioning_mesh_shape: Vec::new(),
            auto_spmd_partitioning_mesh_ids: Vec::new(),

            process_index: 0,
            process_count: 1,

            fdo_profile: Vec::new(),
            debug_options: DebugOptions::default(),
        }
    }
}

impl XlaCompileOptions {
    pub fn single_device(device_ordinal: i64) -> Self {
        Self {
            device_ordinal: Some(device_ordinal),
            device_assignment: Some(DeviceAssignment::single(device_ordinal)),
            ..Self::default()
        }
    }

    pub fn replicated(replicas: i64) -> Self {
        Self {
            num_replicas: replicas.max(1),
            ..Self::default()
        }
    }

    pub fn spmd(partitions: i64) -> Self {
        Self {
            num_partitions: partitions.max(1),
            use_spmd_partitioning: partitions > 1,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_replicas <= 0 {
            return Err("num_replicas must be greater than zero".into());
        }

        if self.num_partitions <= 0 {
            return Err("num_partitions must be greater than zero".into());
        }

        if self.process_count <= 0 {
            return Err("process_count must be greater than zero".into());
        }

        if self.process_index < 0 || self.process_index >= self.process_count {
            return Err(format!(
                "process_index {} is outside process_count {}",
                self.process_index, self.process_count
            ));
        }

        if let Some(assignment) = &self.device_assignment {
            assignment.validate(self.num_replicas, self.num_partitions)?;
        }

        if self.auto_spmd_partitioning_mesh_ids.is_empty()
            != self.auto_spmd_partitioning_mesh_shape.is_empty()
        {
            return Err(
                "auto SPMD mesh shape and mesh ids must either both be set or both be empty"
                    .into(),
            );
        }

        Ok(())
    }
}

/// Compatibility adapter for the earlier Severian XLA pipeline options.
impl From<&crate::pipeline::CompileOptions> for XlaCompileOptions {
    fn from(options: &crate::pipeline::CompileOptions) -> Self {
        let mut debug_options = DebugOptions::default();
        for (name, value) in &options.debug_options {
            debug_options.set_string(name.clone(), value.clone());
        }

        Self {
            parameter_is_tupled_arguments: options.parameter_is_tupled_arguments,
            compile_portable_executable: options.portable_artifact,
            device_ordinal: options.device_ordinal.map(|value| value as i64),
            num_replicas: options.num_replicas as i64,
            num_partitions: options.num_partitions as i64,
            use_spmd_partitioning: options.use_spmd_partitioning,
            debug_options,
            ..Self::default()
        }
    }
}
