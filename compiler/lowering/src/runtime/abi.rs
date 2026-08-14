use severian_hir::ValueType;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeSymbol {
    TaskAwaitUnit,
    ChannelCreate,
    ChannelSendPtrAsync,
    ChannelReceivePtr,
    ChannelSelectPtr,
    MutexCreate,
    MutexLock,
    MutexUnlock,
    RwLockCreate,
    RwLockReadLock,
    RwLockReadUnlock,
    RwLockWriteLock,
    RwLockWriteUnlock,
    SemaphoreCreate,
    SemaphoreAcquire,
    SemaphoreRelease,
}

impl RuntimeSymbol {
    pub const fn name(self) -> &'static str {
        match self {
            Self::TaskAwaitUnit => "__sev_task_await_unit",
            Self::ChannelCreate => "__sev_channel_create",
            Self::ChannelSendPtrAsync => "__sev_channel_send_ptr_async",
            Self::ChannelReceivePtr => "__sev_channel_receive_ptr",
            Self::ChannelSelectPtr => "__sev_channel_select_ptr",
            Self::MutexCreate => "__sev_mutex_create",
            Self::MutexLock => "__sev_mutex_lock",
            Self::MutexUnlock => "__sev_mutex_unlock",
            Self::RwLockCreate => "__sev_rwlock_create",
            Self::RwLockReadLock => "__sev_rwlock_read_lock",
            Self::RwLockReadUnlock => "__sev_rwlock_read_unlock",
            Self::RwLockWriteLock => "__sev_rwlock_write_lock",
            Self::RwLockWriteUnlock => "__sev_rwlock_write_unlock",
            Self::SemaphoreCreate => "__sev_semaphore_create",
            Self::SemaphoreAcquire => "__sev_semaphore_acquire",
            Self::SemaphoreRelease => "__sev_semaphore_release",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredValue {
    pub value: String,
    pub ty: ValueType,
}

impl LoweredValue {
    pub fn new(value: impl Into<String>, ty: ValueType) -> Self {
        Self {
            value: value.into(),
            ty,
        }
    }

    pub fn mlir_type(&self) -> &'static str {
        mlir_type(self.ty)
    }
}

pub fn runtime_declarations(
    task_return_types: impl IntoIterator<Item = ValueType>,
    uses_channels: bool,
    uses_sync: bool,
) -> String {
    let mut output = String::new();

    output.push_str("  llvm.func @__sev_task_await_unit(!llvm.ptr)\n");

    let mut suffixes = std::collections::BTreeSet::new();
    for ty in task_return_types {
        if ty == ValueType::Unit {
            continue;
        }
        let suffix = task_type_suffix(ty);
        if suffixes.insert(suffix) {
            writeln!(
                output,
                "  llvm.func @__sev_task_await_{suffix}(!llvm.ptr) -> {}",
                mlir_type(ty)
            )
            .unwrap();
        }
    }

    if uses_channels {
        output.push_str(concat!(
            "  llvm.func @__sev_channel_create(i64) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_send_ptr_async(!llvm.ptr, !llvm.ptr) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_receive_ptr(!llvm.ptr) -> !llvm.ptr\n",
            "  llvm.func @__sev_channel_select_ptr(!llvm.ptr, i64) -> !llvm.ptr\n",
        ));
    }

    if uses_sync {
        output.push_str(concat!(
            "  llvm.func @__sev_mutex_create() -> !llvm.ptr\n",
            "  llvm.func @__sev_mutex_lock(!llvm.ptr)\n",
            "  llvm.func @__sev_mutex_unlock(!llvm.ptr)\n",
            "  llvm.func @__sev_rwlock_create() -> !llvm.ptr\n",
            "  llvm.func @__sev_rwlock_read_lock(!llvm.ptr)\n",
            "  llvm.func @__sev_rwlock_read_unlock(!llvm.ptr)\n",
            "  llvm.func @__sev_rwlock_write_lock(!llvm.ptr)\n",
            "  llvm.func @__sev_rwlock_write_unlock(!llvm.ptr)\n",
            "  llvm.func @__sev_semaphore_create(i64) -> !llvm.ptr\n",
            "  llvm.func @__sev_semaphore_acquire(!llvm.ptr) -> i1\n",
            "  llvm.func @__sev_semaphore_release(!llvm.ptr)\n",
        ));
    }

    output
}

pub fn mlir_type(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "i1",
        ValueType::Unit => "!llvm.void",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Tensor(_)
        | ValueType::TensorAny
        | ValueType::Channel
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Interface(_)
        | ValueType::Any => "!llvm.ptr",
    }
}

pub fn task_type_suffix(ty: ValueType) -> &'static str {
    match ty {
        ValueType::Int => "i64",
        ValueType::Float => "f64",
        ValueType::Bool => "bool",
        ValueType::Unit => "unit",
        ValueType::String
        | ValueType::List
        | ValueType::Tuple
        | ValueType::Map
        | ValueType::Set
        | ValueType::Tensor(_)
        | ValueType::TensorAny
        | ValueType::Channel
        | ValueType::Function
        | ValueType::Result
        | ValueType::Option
        | ValueType::Interface(_)
        | ValueType::Any => "ptr",
    }
}
