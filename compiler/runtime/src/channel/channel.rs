use std::{
    collections::VecDeque,
    fmt,
    sync::{Arc, Condvar, Mutex},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    Closed,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "channel closed")
    }
}

impl std::error::Error for ChannelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrySendError {
    Full,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    Empty,
    Closed,
}

struct State<T> {
    queue: VecDeque<T>,
    capacity: usize,
    closed: bool,
    senders: usize,
    receivers: usize,
}

struct Inner<T> {
    state: Mutex<State<T>>,
    can_send: Condvar,
    can_recv: Condvar,
}

pub struct Channel<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Channel<T> {
    pub fn bounded(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                queue: VecDeque::with_capacity(capacity),
                capacity,
                closed: false,
                senders: 1,
                receivers: 1,
            }),
            can_send: Condvar::new(),
            can_recv: Condvar::new(),
        });

        (
            Sender {
                inner: Arc::clone(&inner),
            },
            Receiver { inner },
        )
    }

    pub fn unbounded() -> (Sender<T>, Receiver<T>) {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                capacity: usize::MAX,
                closed: false,
                senders: 1,
                receivers: 1,
            }),
            can_send: Condvar::new(),
            can_recv: Condvar::new(),
        });

        (
            Sender {
                inner: Arc::clone(&inner),
            },
            Receiver { inner },
        )
    }
}

pub struct Sender<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), ChannelError> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        while !state.closed && state.queue.len() >= state.capacity {
            state = self
                .inner
                .can_send
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }

        if state.closed || state.receivers == 0 {
            return Err(ChannelError::Closed);
        }

        state.queue.push_back(value);
        self.inner.can_recv.notify_one();
        Ok(())
    }

    pub fn try_send(&self, value: T) -> Result<(), (TrySendError, T)> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        if state.closed || state.receivers == 0 {
            return Err((TrySendError::Closed, value));
        }

        if state.queue.len() >= state.capacity {
            return Err((TrySendError::Full, value));
        }

        state.queue.push_back(value);
        self.inner.can_recv.notify_one();
        Ok(())
    }

    pub fn close(&self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed = true;
        self.inner.can_send.notify_all();
        self.inner.can_recv.notify_all();
    }

    pub fn len(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .queue
            .len()
    }

    pub fn capacity(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .capacity
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.senders += 1;
        drop(state);
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.senders = state.senders.saturating_sub(1);
        if state.senders == 0 {
            state.closed = true;
            self.inner.can_recv.notify_all();
        }
    }
}

pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, ChannelError> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        loop {
            if let Some(value) = state.queue.pop_front() {
                self.inner.can_send.notify_one();
                return Ok(value);
            }

            if state.closed || state.senders == 0 {
                return Err(ChannelError::Closed);
            }

            state = self
                .inner
                .can_recv
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }
    }

    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());

        if let Some(value) = state.queue.pop_front() {
            self.inner.can_send.notify_one();
            return Ok(value);
        }

        if state.closed || state.senders == 0 {
            Err(TryRecvError::Closed)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .queue
            .len()
    }

    pub fn is_closed(&self) -> bool {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed || state.senders == 0
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.receivers += 1;
        drop(state);
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.receivers = state.receivers.saturating_sub(1);
        if state.receivers == 0 {
            state.closed = true;
            self.inner.can_send.notify_all();
        }
    }
}
