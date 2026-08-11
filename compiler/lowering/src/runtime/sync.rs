use super::abi::LoweredValue;
use severian_hir::ValueType;

pub fn emit_mutex_create(result: &str) -> (LoweredValue, String) {
    (
        LoweredValue::new(result, ValueType::Any),
        format!("    {result} = llvm.call @__sev_mutex_create() : () -> !llvm.ptr\n"),
    )
}

pub fn emit_mutex_lock(mutex: &str) -> String {
    format!("    llvm.call @__sev_mutex_lock({mutex}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_mutex_unlock(mutex: &str) -> String {
    format!("    llvm.call @__sev_mutex_unlock({mutex}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_rwlock_create(result: &str) -> (LoweredValue, String) {
    (
        LoweredValue::new(result, ValueType::Any),
        format!("    {result} = llvm.call @__sev_rwlock_create() : () -> !llvm.ptr\n"),
    )
}

pub fn emit_rwlock_read_lock(lock: &str) -> String {
    format!("    llvm.call @__sev_rwlock_read_lock({lock}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_rwlock_read_unlock(lock: &str) -> String {
    format!("    llvm.call @__sev_rwlock_read_unlock({lock}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_rwlock_write_lock(lock: &str) -> String {
    format!("    llvm.call @__sev_rwlock_write_lock({lock}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_rwlock_write_unlock(lock: &str) -> String {
    format!("    llvm.call @__sev_rwlock_write_unlock({lock}) : (!llvm.ptr) -> ()\n")
}

pub fn emit_semaphore_create(result: &str, permits: &str) -> (LoweredValue, String) {
    (
        LoweredValue::new(result, ValueType::Any),
        format!(
            "    {result} = llvm.call @__sev_semaphore_create({permits}) : (i64) -> !llvm.ptr\n"
        ),
    )
}

pub fn emit_semaphore_acquire(result: &str, semaphore: &str) -> (LoweredValue, String) {
    (
        LoweredValue::new(result, ValueType::Bool),
        format!(
            "    {result} = llvm.call @__sev_semaphore_acquire({semaphore}) : (!llvm.ptr) -> i1\n"
        ),
    )
}

pub fn emit_semaphore_release(semaphore: &str) -> String {
    format!("    llvm.call @__sev_semaphore_release({semaphore}) : (!llvm.ptr) -> ()\n")
}
