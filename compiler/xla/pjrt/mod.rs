//! PJRT runtime abstraction.
//!
//! OpenXLA recommends PJRT as the framework/device integration boundary. The
//! exact C ABI is isolated behind `PjrtBackend`; the rest of Severian only sees
//! Rust-owned devices, buffers and executables.

pub mod buffer;
pub mod client;
pub mod device;
pub mod executable;

pub use buffer::{Buffer, BufferId, ElementType, HostBuffer, Shape};
pub use client::{PjrtBackend, PjrtClient, PjrtPlugin};
pub use device::{Device, DeviceId, DeviceKind, MemorySpace};
pub use executable::{
    ExecuteOptions, ExecutionResult, ExecutableId, LoadedExecutable,
};
