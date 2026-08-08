pub mod amd;
pub mod device;
pub mod nvidia;
pub mod target;

pub use device::{GpuDevice, GpuVendor};
pub use target::GpuTarget;

pub fn detect_devices() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    devices.extend(nvidia::detect_devices());
    devices.extend(amd::detect_devices());
    devices
}
