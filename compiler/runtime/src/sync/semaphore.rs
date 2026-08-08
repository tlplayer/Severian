use std::sync::{Arc, Condvar, Mutex};

struct State {
    permits: usize,
    closed: bool,
}

struct Inner {
    state: Mutex<State>,
    available: Condvar,
}

#[derive(Clone)]
pub struct Semaphore {
    inner: Arc<Inner>,
}

impl Semaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    permits,
                    closed: false,
                }),
                available: Condvar::new(),
            }),
        }
    }

    pub fn acquire(&self) -> Option<Permit> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        while state.permits == 0 && !state.closed {
            state = self
                .inner
                .available
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }

        if state.closed {
            return None;
        }

        state.permits -= 1;
        Some(Permit {
            semaphore: self.clone(),
            released: false,
        })
    }

    pub fn try_acquire(&self) -> Option<Permit> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        if state.closed || state.permits == 0 {
            return None;
        }

        state.permits -= 1;
        Some(Permit {
            semaphore: self.clone(),
            released: false,
        })
    }

    pub fn available_permits(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .permits
    }

    pub fn add_permits(&self, permits: usize) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.permits = state.permits.saturating_add(permits);
        self.inner.available.notify_all();
    }

    pub fn close(&self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed = true;
        self.inner.available.notify_all();
    }

    fn release_one(&self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.closed {
            state.permits = state.permits.saturating_add(1);
            self.inner.available.notify_one();
        }
    }
}

pub struct Permit {
    semaphore: Semaphore,
    released: bool,
}

impl Permit {
    pub fn release(mut self) {
        if !self.released {
            self.semaphore.release_one();
            self.released = true;
        }
    }

    pub fn forget(mut self) {
        self.released = true;
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if !self.released {
            self.semaphore.release_one();
            self.released = true;
        }
    }
}
