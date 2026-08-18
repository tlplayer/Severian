//! Compiler-side lowering for calls into the Severian runtime ABI.

pub mod abi;
#[path = "await.rs"]
pub mod await_lowering;
pub mod channels;
pub mod sync;
pub mod tasks;

pub use abi::{mlir_type, runtime_declarations, task_type_suffix, LoweredValue, RuntimeSymbol};
pub use await_lowering::{emit_await, emit_await_many, AwaitLowering};
pub use channels::{
    emit_channel_create, emit_channel_receive, emit_channel_select, emit_channel_send,
    ChannelSelectCase, ChannelSelectLowering,
};
pub use sync::{
    emit_mutex_create, emit_mutex_lock, emit_mutex_unlock, emit_rwlock_create,
    emit_rwlock_read_lock, emit_rwlock_read_unlock, emit_rwlock_write_lock,
    emit_rwlock_write_unlock, emit_semaphore_acquire, emit_semaphore_create,
    emit_semaphore_release,
};
pub use tasks::{
    emit_task_spawn, placement_attributes, task_spawn_declaration, TaskSpawnLowering, TaskSpawnSpec,
};
