mod park;
mod queue;
mod task;
mod wake;
mod worker;

pub use park::{ParkHandle, Parker};
pub use queue::RunQueue;
pub use task::ScheduledTask;
pub use wake::WakeHandle;
pub use worker::{Worker, WorkerId};

use crate::task::{Placement, Task, TaskFn, TaskHandle, TaskState};
use std::{
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
};

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub worker_threads: usize,
    pub global_queue_capacity: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1),
            global_queue_capacity: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SchedulerStats {
    pub spawned: u64,
    pub completed: u64,
    pub steals: u64,
}

pub struct Scheduler {
    queue: Arc<RunQueue>,
    shutdown: Arc<AtomicBool>,
    workers: Mutex<Vec<Worker>>,
    spawned: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
}

impl Scheduler {
    pub fn new(config: RuntimeConfig) -> io::Result<Self> {
        let queue = Arc::new(RunQueue::new(config.global_queue_capacity));
        let shutdown = Arc::new(AtomicBool::new(false));
        let spawned = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));

        let scheduler = Self {
            queue,
            shutdown,
            workers: Mutex::new(Vec::new()),
            spawned,
            completed,
        };

        scheduler.start_workers(config.worker_threads)?;
        Ok(scheduler)
    }

    fn start_workers(&self, count: usize) -> io::Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(|p| p.into_inner());

        for index in 0..count.max(1) {
            workers.push(Worker::spawn(
                WorkerId(index),
                Arc::clone(&self.queue),
                Arc::clone(&self.shutdown),
                Arc::clone(&self.completed),
            )?);
        }

        Ok(())
    }

    pub fn spawn(
        &self,
        name: Option<String>,
        placement: Placement,
        body: TaskFn,
    ) -> TaskHandle {
        let (task, handle) = Task::new(name, placement, body);
        task.set_state(TaskState::Runnable);
        self.spawned.fetch_add(1, Ordering::Relaxed);
        self.queue.push(ScheduledTask::new(task));
        handle
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            spawned: self.spawned.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            steals: 0,
        }
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.queue.close();
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown();
        let workers = self.workers.get_mut().unwrap_or_else(|p| p.into_inner());
        for worker in workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub struct Runtime {
    scheduler: Scheduler,
}

impl Runtime {
    pub fn new(config: RuntimeConfig) -> io::Result<Self> {
        Ok(Self {
            scheduler: Scheduler::new(config)?,
        })
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    pub fn spawn(
        &self,
        name: impl Into<String>,
        body: impl FnOnce(crate::task::TaskContext) -> crate::task::TaskResult + Send + 'static,
    ) -> TaskHandle {
        self.scheduler.spawn(
            Some(name.into()),
            Placement::Default,
            Box::new(body),
        )
    }

    pub fn spawn_on(
        &self,
        name: impl Into<String>,
        placement: Placement,
        body: impl FnOnce(crate::task::TaskContext) -> crate::task::TaskResult + Send + 'static,
    ) -> TaskHandle {
        self.scheduler
            .spawn(Some(name.into()), placement, Box::new(body))
    }
}
