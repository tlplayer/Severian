use super::{handle_from_id, id_from_handle, next_handle_id};
use crate::sync::{Permit, Semaphore};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Condvar, Mutex, OnceLock},
};

struct AbiMutex {
    semaphore: Semaphore,
    permit: Mutex<Option<Permit>>,
}

impl AbiMutex {
    fn new() -> Self {
        Self {
            semaphore: Semaphore::new(1),
            permit: Mutex::new(None),
        }
    }

    fn lock(&self) {
        if let Some(permit) = self.semaphore.acquire() {
            *self
                .permit
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = Some(permit);
        }
    }

    fn unlock(&self) {
        self.permit
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
    }
}

#[derive(Default)]
struct RwState {
    readers: usize,
    writer: bool,
    waiting_writers: usize,
}

struct AbiRwLock {
    state: Mutex<RwState>,
    ready: Condvar,
}

impl AbiRwLock {
    fn new() -> Self {
        Self {
            state: Mutex::new(RwState::default()),
            ready: Condvar::new(),
        }
    }

    fn read_lock(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        while state.writer || state.waiting_writers > 0 {
            state = self.ready.wait(state).unwrap_or_else(|poison| poison.into_inner());
        }
        state.readers += 1;
    }

    fn read_unlock(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.readers = state.readers.saturating_sub(1);
        if state.readers == 0 {
            self.ready.notify_all();
        }
    }

    fn write_lock(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.waiting_writers += 1;
        while state.writer || state.readers > 0 {
            state = self.ready.wait(state).unwrap_or_else(|poison| poison.into_inner());
        }
        state.waiting_writers = state.waiting_writers.saturating_sub(1);
        state.writer = true;
    }

    fn write_unlock(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poison| poison.into_inner());
        state.writer = false;
        self.ready.notify_all();
    }
}

struct AbiSemaphore {
    semaphore: Semaphore,
    held: Mutex<Vec<Permit>>,
}

fn mutexes() -> &'static Mutex<HashMap<usize, AbiMutex>> {
    static VALUES: OnceLock<Mutex<HashMap<usize, AbiMutex>>> = OnceLock::new();
    VALUES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn rwlocks() -> &'static Mutex<HashMap<usize, AbiRwLock>> {
    static VALUES: OnceLock<Mutex<HashMap<usize, AbiRwLock>>> = OnceLock::new();
    VALUES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn semaphores() -> &'static Mutex<HashMap<usize, AbiSemaphore>> {
    static VALUES: OnceLock<Mutex<HashMap<usize, AbiSemaphore>>> = OnceLock::new();
    VALUES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[no_mangle]
pub extern "C" fn __sev_mutex_create() -> *mut c_void {
    let id = next_handle_id();
    mutexes()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, AbiMutex::new());
    handle_from_id(id)
}

#[no_mangle]
pub extern "C" fn __sev_mutex_lock(handle: *mut c_void) {
    if let Some(mutex) = mutexes()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        mutex.lock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_mutex_unlock(handle: *mut c_void) {
    if let Some(mutex) = mutexes()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        mutex.unlock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_rwlock_create() -> *mut c_void {
    let id = next_handle_id();
    rwlocks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(id, AbiRwLock::new());
    handle_from_id(id)
}

#[no_mangle]
pub extern "C" fn __sev_rwlock_read_lock(handle: *mut c_void) {
    if let Some(lock) = rwlocks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        lock.read_lock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_rwlock_read_unlock(handle: *mut c_void) {
    if let Some(lock) = rwlocks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        lock.read_unlock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_rwlock_write_lock(handle: *mut c_void) {
    if let Some(lock) = rwlocks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        lock.write_lock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_rwlock_write_unlock(handle: *mut c_void) {
    if let Some(lock) = rwlocks()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&id_from_handle(handle))
    {
        lock.write_unlock();
    }
}

#[no_mangle]
pub extern "C" fn __sev_semaphore_create(permits: i64) -> *mut c_void {
    let id = next_handle_id();
    let permits = usize::try_from(permits.max(0)).unwrap_or(0);
    semaphores()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(
            id,
            AbiSemaphore {
                semaphore: Semaphore::new(permits),
                held: Mutex::new(Vec::new()),
            },
        );
    handle_from_id(id)
}

#[no_mangle]
pub extern "C" fn __sev_semaphore_acquire(handle: *mut c_void) -> bool {
    let guard = semaphores()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(semaphore) = guard.get(&id_from_handle(handle)) else {
        return false;
    };

    let Some(permit) = semaphore.semaphore.acquire() else {
        return false;
    };

    semaphore
        .held
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .push(permit);
    true
}

#[no_mangle]
pub extern "C" fn __sev_semaphore_release(handle: *mut c_void) {
    let guard = semaphores()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(semaphore) = guard.get(&id_from_handle(handle)) else {
        return;
    };

    semaphore
        .held
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .pop();
}
