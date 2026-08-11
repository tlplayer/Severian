use super::{handle_from_id, id_from_handle, next_handle_id, null_handle};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone)]
pub(crate) enum BoxedValue {
    I64(i64),
    F64(f64),
    Bool(bool),
    Pointer(usize),
    StringPointer(usize),
    CollectionPointer(usize),
}

fn values() -> &'static Mutex<HashMap<usize, BoxedValue>> {
    static VALUES: OnceLock<Mutex<HashMap<usize, BoxedValue>>> = OnceLock::new();
    VALUES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn insert_value(value: BoxedValue) -> *mut c_void {
    let id = next_handle_id();
    values()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, value);
    handle_from_id(id)
}

pub(crate) fn snapshot(handle: *mut c_void) -> Option<BoxedValue> {
    let id = id_from_handle(handle);
    values()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id)
        .cloned()
}

pub(crate) fn clone_handle(handle: *mut c_void) -> *mut c_void {
    match snapshot(handle) {
        Some(value) => insert_value(value),
        None => handle,
    }
}

pub(crate) fn value_equal(left: *mut c_void, right: *mut c_void) -> bool {
    match (snapshot(left), snapshot(right)) {
        (Some(BoxedValue::I64(a)), Some(BoxedValue::I64(b))) => a == b,
        (Some(BoxedValue::F64(a)), Some(BoxedValue::F64(b))) => a == b,
        (Some(BoxedValue::Bool(a)), Some(BoxedValue::Bool(b))) => a == b,
        (Some(BoxedValue::Pointer(a)), Some(BoxedValue::Pointer(b)))
        | (Some(BoxedValue::StringPointer(a)), Some(BoxedValue::StringPointer(b)))
        | (Some(BoxedValue::CollectionPointer(a)), Some(BoxedValue::CollectionPointer(b))) => {
            a == b
        }
        _ => left == right,
    }
}

pub(crate) fn value_less(left: *mut c_void, right: *mut c_void) -> bool {
    match (snapshot(left), snapshot(right)) {
        (Some(BoxedValue::I64(a)), Some(BoxedValue::I64(b))) => a < b,
        (Some(BoxedValue::F64(a)), Some(BoxedValue::F64(b))) => a < b,
        (Some(BoxedValue::Bool(a)), Some(BoxedValue::Bool(b))) => !a && b,
        _ => id_from_handle(left) < id_from_handle(right),
    }
}

#[no_mangle]
pub extern "C" fn __sev_box_i64(value: i64) -> *mut c_void {
    insert_value(BoxedValue::I64(value))
}

#[no_mangle]
pub extern "C" fn __sev_box_f64(value: f64) -> *mut c_void {
    insert_value(BoxedValue::F64(value))
}

#[no_mangle]
pub extern "C" fn __sev_box_bool(value: bool) -> *mut c_void {
    insert_value(BoxedValue::Bool(value))
}

#[no_mangle]
pub extern "C" fn __sev_box_string(value: *mut c_void) -> *mut c_void {
    insert_value(BoxedValue::StringPointer(id_from_handle(value)))
}

#[no_mangle]
pub extern "C" fn __sev_box_collection(value: *mut c_void) -> *mut c_void {
    insert_value(BoxedValue::CollectionPointer(id_from_handle(value)))
}

#[no_mangle]
pub extern "C" fn __sev_box_ptr(value: *mut c_void) -> *mut c_void {
    insert_value(BoxedValue::Pointer(id_from_handle(value)))
}

#[no_mangle]
pub extern "C" fn __sev_unbox_i64(value: *mut c_void) -> i64 {
    match snapshot(value) {
        Some(BoxedValue::I64(value)) => value,
        Some(BoxedValue::Bool(value)) => i64::from(value),
        Some(BoxedValue::F64(value)) => value as i64,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn __sev_unbox_f64(value: *mut c_void) -> f64 {
    match snapshot(value) {
        Some(BoxedValue::F64(value)) => value,
        Some(BoxedValue::I64(value)) => value as f64,
        Some(BoxedValue::Bool(value)) => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn __sev_unbox_bool(value: *mut c_void) -> bool {
    match snapshot(value) {
        Some(BoxedValue::Bool(value)) => value,
        Some(BoxedValue::I64(value)) => value != 0,
        Some(BoxedValue::F64(value)) => value != 0.0,
        Some(BoxedValue::Pointer(value))
        | Some(BoxedValue::StringPointer(value))
        | Some(BoxedValue::CollectionPointer(value)) => value != 0,
        None => !value.is_null(),
    }
}

#[no_mangle]
pub extern "C" fn __sev_unbox_string(value: *mut c_void) -> *mut c_void {
    match snapshot(value) {
        Some(BoxedValue::StringPointer(value)) | Some(BoxedValue::Pointer(value)) => {
            handle_from_id(value)
        }
        _ => null_handle(),
    }
}

#[no_mangle]
pub extern "C" fn __sev_unbox_ptr(value: *mut c_void) -> *mut c_void {
    match snapshot(value) {
        Some(BoxedValue::Pointer(value))
        | Some(BoxedValue::StringPointer(value))
        | Some(BoxedValue::CollectionPointer(value)) => handle_from_id(value),
        None => value,
        _ => null_handle(),
    }
}

#[no_mangle]
pub extern "C" fn __sev_value_equal(left: *mut c_void, right: *mut c_void) -> bool {
    value_equal(left, right)
}

#[no_mangle]
pub extern "C" fn __sev_value_less(left: *mut c_void, right: *mut c_void) -> bool {
    value_less(left, right)
}
