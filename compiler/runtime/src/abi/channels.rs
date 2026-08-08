use super::{
    boxing::__sev_box_i64,
    collections::{collection_from_handles, collection_values},
    handle_from_id, id_from_handle, next_handle_id, null_handle,
    tasks::__sev_task_await_unit,
};
use crate::{
    channel::{select, Channel, Receiver, SelectCase, SelectResult, Sender},
    task::Placement,
    Runtime, RuntimeConfig,
};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Mutex, OnceLock},
    time::Duration,
};

struct AbiChannel {
    sender: Sender<usize>,
    receiver: Receiver<usize>,
}

fn channels() -> &'static Mutex<HashMap<usize, AbiChannel>> {
    static CHANNELS: OnceLock<Mutex<HashMap<usize, AbiChannel>>> = OnceLock::new();
    CHANNELS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new(RuntimeConfig::default())
            .expect("failed to initialize Severian channel runtime")
    })
}

fn channel_receiver(handle: *mut c_void) -> Option<Receiver<usize>> {
    channels()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
        .map(|channel| channel.receiver.clone())
}

fn channel_sender(handle: *mut c_void) -> Option<Sender<usize>> {
    channels()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
        .map(|channel| channel.sender.clone())
}

#[no_mangle]
pub extern "C" fn __sev_channel_create(capacity: i64) -> *mut c_void {
    // The first runtime Channel implementation does not yet have a rendezvous
    // path for capacity=0, so treat zero as a one-slot channel until Codex adds
    // true unbuffered semantics.
    let capacity = usize::try_from(capacity.max(1)).unwrap_or(1);
    let (sender, receiver) = Channel::bounded(capacity);

    let id = next_handle_id();
    channels()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, AbiChannel { sender, receiver });

    handle_from_id(id)
}

#[no_mangle]
pub extern "C" fn __sev_channel_send_ptr_async(
    value: *mut c_void,
    channel: *mut c_void,
) -> *mut c_void {
    let Some(sender) = channel_sender(channel) else {
        return null_handle();
    };

    let value = id_from_handle(value);
    let task = runtime().spawn_on("channel-send", Placement::Default, move |context| {
        context.check_cancelled()?;
        sender
            .send(value)
            .map_err(|error| crate::task::TaskError::Failed(error.to_string()))?;
        Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
    });

    // Reuse the task ABI's registry by spawning through a tiny join adapter
    // would require exposing its private insert function. Keep this handle in a
    // local send-task registry instead.
    insert_send_task(task)
}

fn send_tasks() -> &'static Mutex<HashMap<usize, crate::TaskHandle>> {
    static TASKS: OnceLock<Mutex<HashMap<usize, crate::TaskHandle>>> = OnceLock::new();
    TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn insert_send_task(task: crate::TaskHandle) -> *mut c_void {
    let id = next_handle_id();
    send_tasks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, task);
    handle_from_id(id)
}

#[no_mangle]
pub extern "C" fn __sev_channel_send_await(handle: *mut c_void) {
    let id = id_from_handle(handle);
    if let Some(task) = send_tasks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(&id)
    {
        let _ = task.join();
    } else {
        // If this happens to be a normal task handle, let the normal ABI try it.
        __sev_task_await_unit(handle);
    }
}

#[no_mangle]
pub extern "C" fn __sev_channel_receive_ptr(channel: *mut c_void) -> *mut c_void {
    channel_receiver(channel)
        .and_then(|receiver| receiver.recv().ok())
        .map(handle_from_id)
        .unwrap_or_else(null_handle)
}

#[no_mangle]
pub extern "C" fn __sev_channel_try_receive_ptr(channel: *mut c_void) -> *mut c_void {
    channel_receiver(channel)
        .and_then(|receiver| receiver.try_recv().ok())
        .map(handle_from_id)
        .unwrap_or_else(null_handle)
}

#[no_mangle]
pub extern "C" fn __sev_channel_close(channel: *mut c_void) {
    if let Some(sender) = channel_sender(channel) {
        sender.close();
    }
}

#[no_mangle]
pub extern "C" fn __sev_channel_select_ptr(
    channel_collection: *mut c_void,
    count: i64,
) -> *mut c_void {
    let handles = collection_values(channel_collection);
    let count = usize::try_from(count.max(0))
        .unwrap_or(0)
        .min(handles.len());

    let receivers = handles
        .into_iter()
        .take(count)
        .filter_map(|handle| channel_receiver(handle_from_id(handle)))
        .collect::<Vec<_>>();

    if receivers.is_empty() {
        return null_handle();
    }

    let cases = receivers
        .iter()
        .map(SelectCase::Receive)
        .collect::<Vec<_>>();

    match select(&cases, Duration::from_micros(50)) {
        SelectResult::Received { case_index, value } => {
            collection_from_handles([
                __sev_box_i64(case_index as i64),
                handle_from_id(value),
            ])
        }
        SelectResult::Closed { case_index } => {
            collection_from_handles([
                __sev_box_i64(case_index as i64),
                null_handle(),
            ])
        }
        SelectResult::Default { .. } => null_handle(),
    }
}
