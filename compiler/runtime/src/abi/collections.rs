use super::{
    boxing::{clone_handle, value_equal, value_less},
    handle_from_id, id_from_handle, next_handle_id, null_handle,
};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, Default)]
struct RuntimeCollection {
    values: Vec<usize>,
}

fn collections() -> &'static Mutex<HashMap<usize, RuntimeCollection>> {
    static COLLECTIONS: OnceLock<Mutex<HashMap<usize, RuntimeCollection>>> = OnceLock::new();
    COLLECTIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn insert_collection(collection: RuntimeCollection) -> *mut c_void {
    let id = next_handle_id();
    collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, collection);
    handle_from_id(id)
}

pub(crate) fn collection_values(handle: *mut c_void) -> Vec<usize> {
    collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
        .map(|collection| collection.values.clone())
        .unwrap_or_default()
}

pub(crate) fn collection_from_handles(values: impl IntoIterator<Item = *mut c_void>) -> *mut c_void {
    insert_collection(RuntimeCollection {
        values: values.into_iter().map(id_from_handle).collect(),
    })
}

fn normalize_index(index: i64, len: usize) -> Option<usize> {
    let len = i64::try_from(len).ok()?;
    let normalized = if index < 0 { len.checked_add(index)? } else { index };
    if normalized < 0 || normalized >= len {
        None
    } else {
        usize::try_from(normalized).ok()
    }
}

#[no_mangle]
pub extern "C" fn __sev_collection_new(capacity: i64) -> *mut c_void {
    let capacity = usize::try_from(capacity.max(0)).unwrap_or(0);
    insert_collection(RuntimeCollection {
        values: Vec::with_capacity(capacity),
    })
}

#[no_mangle]
pub extern "C" fn __sev_collection_clone(collection: *mut c_void) -> *mut c_void {
    let values = collection_values(collection)
        .into_iter()
        .map(handle_from_id)
        .map(clone_handle)
        .map(id_from_handle)
        .collect();
    insert_collection(RuntimeCollection { values })
}

#[no_mangle]
pub extern "C" fn __sev_collection_push(collection: *mut c_void, value: *mut c_void) {
    if let Some(collection) = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get_mut(&id_from_handle(collection))
    {
        collection.values.push(id_from_handle(value));
    }
}

#[no_mangle]
pub extern "C" fn __sev_collection_appendleft(collection: *mut c_void, value: *mut c_void) {
    if let Some(collection) = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get_mut(&id_from_handle(collection))
    {
        collection.values.insert(0, id_from_handle(value));
    }
}

#[no_mangle]
pub extern "C" fn __sev_collection_get(collection: *mut c_void, index: i64) -> *mut c_void {
    let guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get(&id_from_handle(collection)) else {
        return null_handle();
    };
    let Some(index) = normalize_index(index, collection.values.len()) else {
        return null_handle();
    };
    handle_from_id(collection.values[index])
}

#[no_mangle]
pub extern "C" fn __sev_collection_set(
    collection: *mut c_void,
    index: i64,
    value: *mut c_void,
) {
    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return;
    };
    let Some(index) = normalize_index(index, collection.values.len()) else {
        return;
    };
    collection.values[index] = id_from_handle(value);
}

#[no_mangle]
pub extern "C" fn __sev_collection_size(collection: *mut c_void) -> i64 {
    collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(collection))
        .map(|collection| i64::try_from(collection.values.len()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn __sev_collection_pop(collection: *mut c_void) -> *mut c_void {
    collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get_mut(&id_from_handle(collection))
        .and_then(|collection| collection.values.pop())
        .map(handle_from_id)
        .unwrap_or_else(null_handle)
}

#[no_mangle]
pub extern "C" fn __sev_collection_pop_at(collection: *mut c_void, index: i64) -> *mut c_void {
    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return null_handle();
    };
    let Some(index) = normalize_index(index, collection.values.len()) else {
        return null_handle();
    };
    handle_from_id(collection.values.remove(index))
}

#[no_mangle]
pub extern "C" fn __sev_collection_last(collection: *mut c_void) -> *mut c_void {
    collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(collection))
        .and_then(|collection| collection.values.last().copied())
        .map(handle_from_id)
        .unwrap_or_else(null_handle)
}

#[no_mangle]
pub extern "C" fn __sev_collection_insert(
    collection: *mut c_void,
    index: i64,
    value: *mut c_void,
) {
    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return;
    };

    let len = collection.values.len() as i64;
    let index = if index < 0 { (len + index).max(0) } else { index };
    let index = usize::try_from(index).unwrap_or(collection.values.len()).min(collection.values.len());
    collection.values.insert(index, id_from_handle(value));
}

#[no_mangle]
pub extern "C" fn __sev_collection_remove(collection: *mut c_void, value: *mut c_void) {
    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return;
    };

    if let Some(index) = collection
        .values
        .iter()
        .position(|candidate| value_equal(handle_from_id(*candidate), value))
    {
        collection.values.remove(index);
    }
}

#[no_mangle]
pub extern "C" fn __sev_collection_extend(
    destination: *mut c_void,
    source: *mut c_void,
) {
    let source_values = collection_values(source);
    if let Some(destination) = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get_mut(&id_from_handle(destination))
    {
        destination.values.extend(source_values);
    }
}

#[no_mangle]
pub extern "C" fn __sev_collection_reversed(collection: *mut c_void) -> *mut c_void {
    let mut values = collection_values(collection);
    values.reverse();
    insert_collection(RuntimeCollection { values })
}

#[no_mangle]
pub extern "C" fn __sev_collection_equal(
    left: *mut c_void,
    right: *mut c_void,
) -> bool {
    let left = collection_values(left);
    let right = collection_values(right);

    left.len() == right.len()
        && left
            .into_iter()
            .zip(right)
            .all(|(left, right)| value_equal(handle_from_id(left), handle_from_id(right)))
}

#[no_mangle]
pub extern "C" fn __sev_collection_sorted(collection: *mut c_void) -> *mut c_void {
    let mut values = collection_values(collection);
    values.sort_by(|left, right| {
        let left = handle_from_id(*left);
        let right = handle_from_id(*right);
        if value_equal(left, right) {
            std::cmp::Ordering::Equal
        } else if value_less(left, right) {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });
    insert_collection(RuntimeCollection { values })
}

#[no_mangle]
pub extern "C" fn __sev_collection_sorted_reverse(
    collection: *mut c_void,
    reverse: bool,
) -> *mut c_void {
    let sorted = __sev_collection_sorted(collection);
    if !reverse {
        return sorted;
    }
    __sev_collection_reversed(sorted)
}

#[no_mangle]
pub extern "C" fn __sev_collection_slice(
    collection: *mut c_void,
    start: i64,
    end: i64,
    step: i64,
) -> *mut c_void {
    let values = collection_values(collection);
    if step == 0 {
        return insert_collection(RuntimeCollection::default());
    }

    let len = values.len() as i64;
    let mut index = if start < 0 { len + start } else { start };
    let end = if end < 0 { len + end } else { end };
    let mut result = Vec::new();

    if step > 0 {
        index = index.max(0);
        let stop = end.min(len);
        while index < stop {
            if let Some(value) = values.get(index as usize) {
                result.push(*value);
            }
            index += step;
        }
    } else {
        index = index.min(len.saturating_sub(1));
        let stop = end.max(-1);
        while index > stop {
            if index >= 0 {
                if let Some(value) = values.get(index as usize) {
                    result.push(*value);
                }
            }
            index += step;
        }
    }

    insert_collection(RuntimeCollection { values: result })
}

#[no_mangle]
pub extern "C" fn __sev_collection_any(collection: *mut c_void) -> bool {
    collection_values(collection)
        .into_iter()
        .any(|value| crate::abi::boxing::__sev_unbox_bool(handle_from_id(value)))
}

#[no_mangle]
pub extern "C" fn __sev_collection_all(collection: *mut c_void) -> bool {
    collection_values(collection)
        .into_iter()
        .all(|value| crate::abi::boxing::__sev_unbox_bool(handle_from_id(value)))
}

#[no_mangle]
pub extern "C" fn __sev_collection_heap_push(
    collection: *mut c_void,
    value: *mut c_void,
) {
    __sev_collection_push(collection, value);

    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return;
    };

    collection.values.sort_by(|left, right| {
        let left = handle_from_id(*left);
        let right = handle_from_id(*right);
        if value_equal(left, right) {
            std::cmp::Ordering::Equal
        } else if value_less(left, right) {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });
}

#[no_mangle]
pub extern "C" fn __sev_collection_heap_pop(collection: *mut c_void) -> *mut c_void {
    let mut guard = collections()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(collection) = guard.get_mut(&id_from_handle(collection)) else {
        return null_handle();
    };
    if collection.values.is_empty() {
        null_handle()
    } else {
        handle_from_id(collection.values.remove(0))
    }
}
