use std::{
    collections::{HashMap, VecDeque},
    sync::{Condvar, Mutex},
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PollKey(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interest {
    pub readable: bool,
    pub writable: bool,
}

impl Interest {
    pub const READABLE: Self = Self {
        readable: true,
        writable: false,
    };

    pub const WRITABLE: Self = Self {
        readable: false,
        writable: true,
    };

    pub const READ_WRITE: Self = Self {
        readable: true,
        writable: true,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollEvent {
    pub key: PollKey,
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
}

struct State {
    registrations: HashMap<PollKey, Interest>,
    ready: VecDeque<PollEvent>,
    closed: bool,
}

/// Runtime-facing polling abstraction.
///
/// This implementation provides the registration/wakeup semantics without
/// committing Severian to epoll/kqueue/io_uring. Platform backends can feed
/// readiness events through `notify`.
pub struct Poller {
    state: Mutex<State>,
    available: Condvar,
}

impl Poller {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                registrations: HashMap::new(),
                ready: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    pub fn register(&self, key: PollKey, interest: Interest) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .registrations
            .insert(key, interest);
    }

    pub fn reregister(&self, key: PollKey, interest: Interest) {
        self.register(key, interest);
    }

    pub fn deregister(&self, key: PollKey) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .registrations
            .remove(&key);
    }

    pub fn notify(&self, event: PollEvent) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        let Some(interest) = state.registrations.get(&event.key).copied() else {
            return;
        };

        let relevant = (interest.readable && event.readable)
            || (interest.writable && event.writable)
            || event.error;

        if relevant {
            state.ready.push_back(event);
            self.available.notify_one();
        }
    }

    pub fn poll(&self, timeout: Option<Duration>) -> Vec<PollEvent> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        if state.ready.is_empty() && !state.closed {
            state = match timeout {
                Some(timeout) => {
                    let (state, _) = self
                        .available
                        .wait_timeout(state, timeout)
                        .unwrap_or_else(|p| p.into_inner());
                    state
                }
                None => self
                    .available
                    .wait(state)
                    .unwrap_or_else(|p| p.into_inner()),
            };
        }

        state.ready.drain(..).collect()
    }

    pub fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed = true;
        self.available.notify_all();
    }
}

impl Default for Poller {
    fn default() -> Self {
        Self::new()
    }
}
