//! PJRT runtime abstraction.
//!
//! OpenXLA recommends PJRT as the framework/device integration boundary. The
//! exact C ABI is isolated in crate-private modules; the public wrappers own
//! the raw buffers and loaded executables they represent.

pub mod buffer;
pub mod client;
pub mod device;
pub mod executable;

// Raw ABI modules remain below the owned Rust API above. The similarly named
// modules are intentional layers: `device`/`buffer`/`executable` are stable
// compiler-facing models, while these modules own PJRT pointers and calls.
pub(crate) mod api;
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

pub use buffer::{Buffer, ElementType, HostBuffer, Shape};
pub use client::{PjrtClient, PjrtPlugin};
pub use device::{Device, DeviceKind, MemorySpace};
pub use executable::LoadedExecutable;
