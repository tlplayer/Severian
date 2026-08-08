use crate::task::{Task, TaskId};
use std::sync::Arc;

#[derive(Clone)]
pub struct ScheduledTask {
    pub task: Arc<Task>,
}

impl ScheduledTask {
    pub fn new(task: Arc<Task>) -> Self {
        Self { task }
    }

    pub fn id(&self) -> TaskId {
        self.task.id
    }
}
