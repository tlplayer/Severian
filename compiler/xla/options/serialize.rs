//! Minimal protobuf wire serializer for OpenXLA CompileOptionsProto.
//!
//! This avoids adding a protobuf dependency to the first XLA integration while
//! still producing a real CompileOptionsProto. Only fields represented by
//! Severian's public option structs are emitted.
//!
//! Current wire schema:
//! CompileOptionsProto:
//!   2 parameter_is_tupled_arguments
//!   3 executable_build_options
//!   4 compile_portable_executable
//!   5 profile_version
//!   6 serialized_multi_slice_config
//!   7 env_option_overrides (map entry)
//!   9 allow_in_place_mlir_modification
//!   11 compiler_variant
//!
//! ExecutableBuildOptionsProto:
//!   1 device_ordinal
//!   3 debug_options (not emitted; intentionally unstable)
//!   4 num_replicas
//!   5 num_partitions
//!   6 use_spmd_partitioning
//!   7 use_auto_spmd_partitioning
//!   8 deduplicate_hlo
//!   9 device_assignment
//!   10 alias_passthrough_params
//!   11 run_backend_only
//!   12 allow_spmd_sharding_propagation_to_output
//!   14 fdo_profile
//!   15 device_memory_size
//!   16 auto_spmd_partitioning_mesh_shape
//!   17 auto_spmd_partitioning_mesh_ids
//!   18 allow_spmd_sharding_propagation_to_parameters
//!   19 use_shardy_partitioner
//!   20 exec_time_optimization_effort
//!   21 memory_fitting_effort
//!   22 process_index
//!   23 process_count
//!   24 optimization_level
//!   25 memory_fitting_level

use super::{
    compile::XlaCompileOptions,
    debug::DebugOptionValue,
    device_assignment::DeviceAssignment,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializeError(pub String);

impl fmt::Display for SerializeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SerializeError {}

pub fn serialize_compile_options(
    options: &XlaCompileOptions,
) -> Result<Vec<u8>, SerializeError> {
    options
        .validate()
        .map_err(SerializeError)?;

    let mut output = Vec::new();

    if options.parameter_is_tupled_arguments {
        field_bool(&mut output, 2, true);
    }

    // OpenXLA CompileOptions::ToProto always materializes
    // executable_build_options, including its non-zero/-1 defaults.
    let build = serialize_build_options(options)?;
    field_bytes(&mut output, 3, &build);

    if options.compile_portable_executable {
        field_bool(&mut output, 4, true);
    }

    if options.profile_version != 0 {
        field_i64(&mut output, 5, options.profile_version);
    }

    if !options.serialized_multi_slice_config.is_empty() {
        field_bytes(
            &mut output,
            6,
            &options.serialized_multi_slice_config,
        );
    }

    // map<string, OptionOverrideProto> env_option_overrides = 7
    for (name, value) in options.debug_options.values() {
        let entry = serialize_override_entry(name, value);
        field_bytes(&mut output, 7, &entry);
    }

    if options.allow_in_place_mlir_modification {
        field_bool(&mut output, 9, true);
    }

    if let Some(variant) = &options.compiler_variant {
        field_bytes(&mut output, 11, variant.as_bytes());
    }

    Ok(output)
}

fn serialize_build_options(
    options: &XlaCompileOptions,
) -> Result<Vec<u8>, SerializeError> {
    let mut output = Vec::new();

    // These are explicitly set by OpenXLA's ExecutableBuildOptions::ToProto.
    field_i64(&mut output, 1, options.device_ordinal.unwrap_or(-1));
    field_i64(&mut output, 4, options.num_replicas);
    field_i64(&mut output, 5, options.num_partitions);

    if options.use_spmd_partitioning {
        field_bool(&mut output, 6, true);
    }

    if options.use_auto_spmd_partitioning {
        field_bool(&mut output, 7, true);
    }

    if options.deduplicate_hlo {
        field_bool(&mut output, 8, true);
    }

    if let Some(assignment) = &options.device_assignment {
        let assignment = serialize_device_assignment(assignment);
        field_bytes(&mut output, 9, &assignment);
    }

    if options.alias_passthrough_params {
        field_bool(&mut output, 10, true);
    }

    if options.run_backend_only {
        field_bool(&mut output, 11, true);
    }

    for &value in &options.allow_spmd_sharding_propagation_to_output {
        field_bool(&mut output, 12, value);
    }

    if !options.fdo_profile.is_empty() {
        field_bytes(&mut output, 14, &options.fdo_profile);
    }

    if let Some(bytes) = options.device_memory_size {
        field_i64(&mut output, 15, bytes);
    }

    for &dimension in &options.auto_spmd_partitioning_mesh_shape {
        field_i64(&mut output, 16, dimension);
    }

    for &device in &options.auto_spmd_partitioning_mesh_ids {
        field_i64(&mut output, 17, device);
    }

    for &value in &options.allow_spmd_sharding_propagation_to_parameters {
        field_bool(&mut output, 18, value);
    }

    if options.use_shardy_partitioner {
        field_bool(&mut output, 19, true);
    }

    if let Some(value) = options.execution_time_effort.relative() {
        field_f32(&mut output, 20, value);
    }

    if let Some(value) = options.memory_fitting_effort.relative() {
        field_f32(&mut output, 21, value);
    }

    if options.process_index != 0 {
        field_i64(&mut output, 22, options.process_index);
    }

    // OpenXLA's default is one process and ToProto explicitly sets it.
    field_i64(&mut output, 23, options.process_count);

    let optimization_level = options.optimization_level.protobuf_value();
    if optimization_level != 0 {
        field_i64(&mut output, 24, optimization_level);
    }

    let memory_fitting_level = options.memory_fitting_level.protobuf_value();
    if memory_fitting_level != 0 {
        field_i64(&mut output, 25, memory_fitting_level);
    }

    Ok(output)
}

fn serialize_device_assignment(assignment: &DeviceAssignment) -> Vec<u8> {
    // DeviceAssignmentProto:
    // int32 replica_count = 1
    // int32 computation_count = 2
    // repeated ComputationDevice computation_devices = 3
    // ComputationDevice: repeated int64 replica_device_ids = 1
    let mut output = Vec::new();

    field_i64(&mut output, 1, assignment.replicas() as i64);
    field_i64(&mut output, 2, assignment.partitions() as i64);

    for partition in assignment.devices() {
        let mut computation = Vec::new();

        // Repeated primitive fields are valid in unpacked form in protobuf.
        for &device in partition {
            field_i64(&mut computation, 1, device);
        }

        field_bytes(&mut output, 3, &computation);
    }

    output
}

fn serialize_override_entry(
    name: &str,
    value: &DebugOptionValue,
) -> Vec<u8> {
    // Map entry:
    // string key = 1
    // OptionOverrideProto value = 2
    let mut entry = Vec::new();
    field_bytes(&mut entry, 1, name.as_bytes());

    let mut override_value = Vec::new();
    match value {
        // OptionOverrideProto oneof field numbers from the current schema.
        DebugOptionValue::String(value) => {
            field_bytes(&mut override_value, 1, value.as_bytes());
        }
        DebugOptionValue::Bool(value) => {
            field_bool(&mut override_value, 2, *value);
        }
        DebugOptionValue::Integer(value) => {
            field_i64(&mut override_value, 3, *value);
        }
        DebugOptionValue::Double(value) => {
            field_f64(&mut override_value, 4, *value);
        }
    }

    field_bytes(&mut entry, 2, &override_value);
    entry
}

fn field_key(output: &mut Vec<u8>, field: u32, wire_type: u8) {
    varint(output, ((field as u64) << 3) | wire_type as u64);
}

fn field_bool(output: &mut Vec<u8>, field: u32, value: bool) {
    field_key(output, field, 0);
    varint(output, u64::from(value));
}

fn field_i64(output: &mut Vec<u8>, field: u32, value: i64) {
    field_key(output, field, 0);
    varint(output, value as u64);
}

fn field_bytes(output: &mut Vec<u8>, field: u32, value: &[u8]) {
    field_key(output, field, 2);
    varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn field_f32(output: &mut Vec<u8>, field: u32, value: f32) {
    field_key(output, field, 5);
    output.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn field_f64(output: &mut Vec<u8>, field: u32, value: f64) {
    field_key(output, field, 1);
    output.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_include_build_options() {
        let bytes = serialize_compile_options(&XlaCompileOptions::default()).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x1a);
    }

    #[test]
    fn single_device_has_build_options() {
        let bytes = serialize_compile_options(&XlaCompileOptions::single_device(3)).unwrap();
        assert!(!bytes.is_empty());
        // CompileOptionsProto executable_build_options field key = 3 << 3 | 2.
        assert_eq!(bytes[0], 0x1a);
    }
}
