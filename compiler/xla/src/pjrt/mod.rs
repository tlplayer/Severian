//! PJRT runtime abstraction.
//!
//! OpenXLA recommends PJRT as the framework/device integration boundary. The
//! exact C ABI is isolated behind `PjrtBackend`; the rest of Severian only sees
//! Rust-owned devices, buffers and executables.

pub mod buffer;
pub mod client;
pub mod device;
pub mod executable;

// Raw ABI modules remain below the owned Rust API above. The similarly named
// modules are intentional layers: `device`/`buffer`/`executable` are stable
// compiler-facing models, while these modules own PJRT pointers and calls.
pub(crate) mod api;
pub(crate) mod assignment;
pub(crate) mod compile;
pub(crate) mod devices;
pub(crate) mod error;
pub(crate) mod events;
pub(crate) mod execute;
pub(crate) mod host_buffer;
pub(crate) mod memory;
pub(crate) mod platform;
pub(crate) mod plugin;
pub(crate) mod topology;

pub use buffer::{Buffer, BufferId, ElementType, HostBuffer, Shape};
pub use client::{PjrtBackend, PjrtClient, PjrtPlugin};
pub use device::{Device, DeviceId, DeviceKind, MemorySpace};
pub use executable::{
    ExecuteOptions, ExecutionResult, ExecutableId, LoadedExecutable,
};
