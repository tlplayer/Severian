use std::collections::BTreeMap;

/// Value type of CompileOptionsProto.env_option_overrides.
///
/// OpenXLA deliberately models these as a protobuf oneof with string, bool,
/// int64 and double variants.
#[derive(Debug, Clone, PartialEq)]
pub enum DebugOptionValue {
    String(String),
    Bool(bool),
    Integer(i64),
    Double(f64),
}

#[derive(Debug, Clone, Default)]
pub struct DebugOptions {
    values: BTreeMap<String, DebugOptionValue>,
}

impl DebugOptions {
    pub fn values(&self) -> &BTreeMap<String, DebugOptionValue> {
        &self.values
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn set(
        &mut self,
        name: impl Into<String>,
        value: DebugOptionValue,
    ) -> &mut Self {
        self.values.insert(name.into(), value);
        self
    }

    pub fn set_string(
        &mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> &mut Self {
        self.set(name, DebugOptionValue::String(value.into()))
    }

    pub fn set_bool(
        &mut self,
        name: impl Into<String>,
        value: bool,
    ) -> &mut Self {
        self.set(name, DebugOptionValue::Bool(value))
    }

    pub fn set_integer(
        &mut self,
        name: impl Into<String>,
        value: i64,
    ) -> &mut Self {
        self.set(name, DebugOptionValue::Integer(value))
    }

    pub fn set_double(
        &mut self,
        name: impl Into<String>,
        value: f64,
    ) -> &mut Self {
        self.set(name, DebugOptionValue::Double(value))
    }

    pub fn remove(&mut self, name: &str) -> Option<DebugOptionValue> {
        self.values.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&DebugOptionValue> {
        self.values.get(name)
    }

    /// Common XLA dump helper. This is stored as an environment-option
    /// override, keeping Severian independent from DebugOptions' very large,
    /// intentionally unstable protobuf surface.
    pub fn dump_to(mut self, directory: impl Into<String>) -> Self {
        self.set_string("xla_dump_to", directory);
        self
    }

    pub fn disable_hlo_passes(mut self, passes: impl Into<String>) -> Self {
        self.set_string("xla_disable_hlo_passes", passes);
        self
    }

    pub fn enable_gpu_triton_gemm(mut self, enabled: bool) -> Self {
        self.set_bool("xla_gpu_triton_gemm_any", enabled);
        self
    }
}
