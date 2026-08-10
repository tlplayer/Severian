use super::{
    buffer::Buffer,
    device::{Device, DeviceId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExecutableId(pub u64);

#[derive(Debug, Clone)]
pub struct LoadedExecutable {
    pub id: ExecutableId,
    pub name: String,
    pub platform: String,
    pub num_replicas: usize,
    pub num_partitions: usize,
    pub addressable_devices: Vec<Device>,
}

#[derive(Debug, Clone)]
pub struct ExecuteOptions {
    pub launch_id: u64,
    pub device: Option<DeviceId>,
    pub untuple_result: bool,
    pub strict_shape_checking: bool,
}

impl Default for ExecuteOptions {
    fn default() -> Self {
        Self {
            launch_id: 0,
            device: None,
            untuple_result: true,
            strict_shape_checking: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionResult {
    pub outputs: Vec<Buffer>,
    pub device: Option<DeviceId>,
    pub complete: bool,
}
