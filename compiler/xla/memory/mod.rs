mod buffer_assignment;
mod buffer_reuse;

pub use buffer_assignment::{
    assign_function_buffers, BufferAssignment, BufferId, BufferPlan, ValueLifetime,
};
pub use buffer_reuse::{apply_reuse, ReuseDecision};
