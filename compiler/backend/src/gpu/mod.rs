pub mod amd;
pub mod nvidia;
pub mod spirv;

pub use amd::{compile_rocm, detect_amd_gpu_chip, lower_to_rocdl};
pub use nvidia::{compile_cuda, detect_nvidia_gpu_architecture, lower_to_nvvm};
pub use spirv::{lower_to_spirv, SpirvTarget};
