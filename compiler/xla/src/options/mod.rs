//! XLA/PJRT compile and execution options.
//!
//! These types are Severian-owned. `serialize` is the only module that knows
//! the wire field numbers of OpenXLA's CompileOptionsProto.

pub mod compile;
pub mod debug;
pub mod device_assignment;
pub mod execution;
pub mod serialize;

pub use compile::{EffortLevel, OptimizationEffort, XlaCompileOptions};
pub use debug::{DebugOptionValue, DebugOptions};
pub use device_assignment::DeviceAssignment;
pub use execution::XlaExecutionOptions;
pub use serialize::{serialize_compile_options, SerializeError};
