mod buffer_assignment;
mod buffer_reuse;

pub use buffer_assignment::{
    assign_function_buffers, BufferAssignment, BufferPlan, BufferSlotId, ValueLifetime,
};
pub use buffer_reuse::{apply_reuse, ReuseDecision};
