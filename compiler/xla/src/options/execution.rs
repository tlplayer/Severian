/// Runtime execution knobs independent of compilation.
///
/// These correspond to the useful subset of PJRT_ExecuteOptions and are kept
/// separate from XLA CompileOptionsProto because execution options are passed
/// directly through the C ABI, not protobuf serialized.
#[derive(Debug, Clone)]
pub struct XlaExecutionOptions {
    pub launch_id: i32,
    pub non_donatable_input_indices: Vec<i64>,
    pub device_ordinal: Option<usize>,
    pub untuple_result: bool,
    pub strict_shape_checking: bool,
}

impl Default for XlaExecutionOptions {
    fn default() -> Self {
        Self {
            launch_id: 0,
            non_donatable_input_indices: Vec::new(),
            device_ordinal: None,
            untuple_result: true,
            strict_shape_checking: true,
        }
    }
}

impl XlaExecutionOptions {
    pub fn with_launch_id(mut self, launch_id: i32) -> Self {
        self.launch_id = launch_id;
        self
    }

    pub fn on_device(mut self, ordinal: usize) -> Self {
        self.device_ordinal = Some(ordinal);
        self
    }

    pub fn preserve_input(mut self, index: usize) -> Self {
        let index = index as i64;
        if !self.non_donatable_input_indices.contains(&index) {
            self.non_donatable_input_indices.push(index);
        }
        self
    }

    pub fn preserve_all_inputs(mut self, input_count: usize) -> Self {
        self.non_donatable_input_indices =
            (0..input_count).map(|index| index as i64).collect();
        self
    }
}

impl From<&crate::pjrt::executable::ExecuteOptions> for XlaExecutionOptions {
    fn from(options: &crate::pjrt::executable::ExecuteOptions) -> Self {
        Self {
            launch_id: options.launch_id.min(i32::MAX as u64) as i32,
            non_donatable_input_indices: Vec::new(),
            device_ordinal: options.device.map(|device| device.0),
            untuple_result: options.untuple_result,
            strict_shape_checking: options.strict_shape_checking,
        }
    }
}
