use super::{handle_from_id, id_from_handle, next_handle_id, null_handle};
use crate::{
    task::{Placement, TaskError, TaskHandle},
    Runtime, RuntimeConfig,
};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, Copy)]
enum TaskValue {
    Unit,
    I64(i64),
    F64(f64),
    Bool(bool),
    Pointer(usize),
}

struct TaskRecord {
    handle: Option<TaskHandle>,
}

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new(RuntimeConfig::default())
            .expect("failed to initialize Severian runtime")
    })
}

fn tasks() -> &'static Mutex<HashMap<usize, TaskRecord>> {
    static TASKS: OnceLock<Mutex<HashMap<usize, TaskRecord>>> = OnceLock::new();
    TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn insert_task(handle: TaskHandle) -> *mut c_void {
    let id = next_handle_id();
    tasks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(
            id,
            TaskRecord {
                handle: Some(handle),
            },
        );
    handle_from_id(id)
}

fn join_task(handle: *mut c_void) -> Option<TaskValue> {
    let id = id_from_handle(handle);
    let mut record = tasks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(&id)?;

    let task = record.handle.take()?;
    match task.join() {
        Ok(value) => value.downcast::<TaskValue>().ok().map(|value| *value),
        Err(TaskError::Cancelled | TaskError::Panicked | TaskError::Failed(_)) => None,
    }
}

fn spawn_value(
    name: &'static str,
    body: impl FnOnce() -> TaskValue + Send + 'static,
) -> *mut c_void {
    let handle = runtime().spawn_on(name, Placement::Default, move |context| {
        context.check_cancelled()?;
        Ok(Box::new(body()) as Box<dyn std::any::Any + Send>)
    });
    insert_task(handle)
}

pub type PtrTaskEntry = extern "C" fn(*mut c_void) -> *mut c_void;
pub type I64TaskEntry = extern "C" fn(*mut c_void) -> i64;
pub type F64TaskEntry = extern "C" fn(*mut c_void) -> f64;
pub type BoolTaskEntry = extern "C" fn(*mut c_void) -> bool;
pub type UnitTaskEntry = extern "C" fn(*mut c_void);

#[no_mangle]
pub extern "C" fn __sev_task_spawn_ptr(
    entry: PtrTaskEntry,
    context: *mut c_void,
) -> *mut c_void {
    let context = id_from_handle(context);
    spawn_value("abi-ptr-task", move || {
        TaskValue::Pointer(id_from_handle(entry(handle_from_id(context))))
    })
}

#[no_mangle]
pub extern "C" fn __sev_task_spawn_i64(
    entry: I64TaskEntry,
    context: *mut c_void,
) -> *mut c_void {
    let context = id_from_handle(context);
    spawn_value("abi-i64-task", move || {
        TaskValue::I64(entry(handle_from_id(context)))
    })
}

#[no_mangle]
pub extern "C" fn __sev_task_spawn_f64(
    entry: F64TaskEntry,
    context: *mut c_void,
) -> *mut c_void {
    let context = id_from_handle(context);
    spawn_value("abi-f64-task", move || {
        TaskValue::F64(entry(handle_from_id(context)))
    })
}

#[no_mangle]
pub extern "C" fn __sev_task_spawn_bool(
    entry: BoolTaskEntry,
    context: *mut c_void,
) -> *mut c_void {
    let context = id_from_handle(context);
    spawn_value("abi-bool-task", move || {
        TaskValue::Bool(entry(handle_from_id(context)))
    })
}

#[no_mangle]
pub extern "C" fn __sev_task_spawn_unit(
    entry: UnitTaskEntry,
    context: *mut c_void,
) -> *mut c_void {
    let context = id_from_handle(context);
    spawn_value("abi-unit-task", move || {
        entry(handle_from_id(context));
        TaskValue::Unit
    })
}

#[no_mangle]
pub extern "C" fn __sev_task_await_unit(handle: *mut c_void) {
    let _ = join_task(handle);
}

#[no_mangle]
pub extern "C" fn __sev_task_await_i64(handle: *mut c_void) -> i64 {
    match join_task(handle) {
        Some(TaskValue::I64(value)) => value,
        Some(TaskValue::Bool(value)) => i64::from(value),
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn __sev_task_await_f64(handle: *mut c_void) -> f64 {
    match join_task(handle) {
        Some(TaskValue::F64(value)) => value,
        Some(TaskValue::I64(value)) => value as f64,
        _ => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn __sev_task_await_bool(handle: *mut c_void) -> bool {
    match join_task(handle) {
        Some(TaskValue::Bool(value)) => value,
        Some(TaskValue::I64(value)) => value != 0,
        _ => false,
    }
}

#[no_mangle]
pub extern "C" fn __sev_task_await_ptr(handle: *mut c_void) -> *mut c_void {
    match join_task(handle) {
        Some(TaskValue::Pointer(value)) => handle_from_id(value),
        _ => null_handle(),
    }
}

#[no_mangle]
pub extern "C" fn __sev_task_cancel(handle: *mut c_void) -> bool {
    let id = id_from_handle(handle);
    let guard = tasks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(record) = guard.get(&id) else {
        return false;
    };
    let Some(task) = record.handle.as_ref() else {
        return false;
    };
    task.cancel();
    true
}

/// Compiler-generated spawn wrappers can use this macro when each Severian
/// function gets a concrete `__sev_task_spawn_<function>` symbol.
///
/// Example generated glue:
///
/// ```ignore
/// severian_task_spawn_wrapper_i64!(
///     __sev_task_spawn_work,
///     __sev_source_work
/// );
/// ```
#[macro_export]
macro_rules! severian_task_spawn_wrapper_i64 {
    ($spawn:ident, $entry:path) => {
        #[no_mangle]
        pub extern "C" fn $spawn(argument: i64) -> *mut std::ffi::c_void {
            extern "C" fn thunk(context: *mut std::ffi::c_void) -> i64 {
                let value = context as usize as i64;
                $entry(value)
            }
            $crate::abi::tasks::__sev_task_spawn_i64(
                thunk,
                argument as usize as *mut std::ffi::c_void,
            )
        }
    };
}
