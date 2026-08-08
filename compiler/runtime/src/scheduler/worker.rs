use super::RunQueue;
use std::{
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(pub usize);

pub struct Worker {
    id: WorkerId,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(
        id: WorkerId,
        queue: Arc<RunQueue>,
        shutdown: Arc<AtomicBool>,
        completed: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        let name = format!("severian-worker-{}", id.0);
        let handle = thread::Builder::new().name(name).spawn(move || {
            while !shutdown.load(Ordering::Acquire) {
                let Some(scheduled) = queue.pop() else {
                    break;
                };

                scheduled.task.run();
                completed.fetch_add(1, Ordering::Relaxed);
            }
        })?;

        Ok(Self {
            id,
            handle: Some(handle),
        })
    }

    pub fn id(&self) -> WorkerId {
        self.id
    }

    pub fn join(mut self) -> thread::Result<()> {
        if let Some(handle) = self.handle.take() {
            handle.join()
        } else {
            Ok(())
        }
    }
}
