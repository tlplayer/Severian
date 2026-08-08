use super::ScheduledTask;
use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

struct QueueState {
    tasks: VecDeque<ScheduledTask>,
    closed: bool,
}

pub struct RunQueue {
    capacity: Option<usize>,
    state: Mutex<QueueState>,
    available: Condvar,
    space: Condvar,
}

impl RunQueue {
    pub fn new(capacity: Option<usize>) -> Self {
        Self {
            capacity,
            state: Mutex::new(QueueState {
                tasks: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
            space: Condvar::new(),
        }
    }

    pub fn push(&self, task: ScheduledTask) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        while !state.closed
            && self
                .capacity
                .is_some_and(|capacity| state.tasks.len() >= capacity)
        {
            state = self.space.wait(state).unwrap_or_else(|p| p.into_inner());
        }

        if state.closed {
            return false;
        }

        state.tasks.push_back(task);
        self.available.notify_one();
        true
    }

    pub fn try_push(&self, task: ScheduledTask) -> Result<(), ScheduledTask> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        if state.closed
            || self
                .capacity
                .is_some_and(|capacity| state.tasks.len() >= capacity)
        {
            return Err(task);
        }

        state.tasks.push_back(task);
        self.available.notify_one();
        Ok(())
    }

    pub fn pop(&self) -> Option<ScheduledTask> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());

        loop {
            if let Some(task) = state.tasks.pop_front() {
                self.space.notify_one();
                return Some(task);
            }

            if state.closed {
                return None;
            }

            state = self
                .available
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }
    }

    pub fn try_pop(&self) -> Option<ScheduledTask> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let task = state.tasks.pop_front();
        if task.is_some() {
            self.space.notify_one();
        }
        task
    }

    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .tasks
            .len()
    }

    pub fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed = true;
        self.available.notify_all();
        self.space.notify_all();
    }
}
