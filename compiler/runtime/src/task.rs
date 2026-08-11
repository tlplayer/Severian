use std::{
    any::Any,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Condvar, Mutex,
    },
};

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub u64);

impl TaskId {
    pub fn next() -> Self {
        Self(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    Default,
    Local,
    Gpu,
    Simd,
    Simt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    Created = 0,
    Runnable = 1,
    Running = 2,
    Parked = 3,
    Completed = 4,
    Cancelled = 5,
    Failed = 6,
}

impl TaskState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Created,
            1 => Self::Runnable,
            2 => Self::Running,
            3 => Self::Parked,
            4 => Self::Completed,
            5 => Self::Cancelled,
            6 => Self::Failed,
            _ => Self::Failed,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskError {
    Cancelled,
    Panicked,
    Failed(String),
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "task cancelled"),
            Self::Panicked => write!(f, "task panicked"),
            Self::Failed(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for TaskError {}

pub type TaskResult = Result<Box<dyn Any + Send>, TaskError>;
pub type TaskFn = Box<dyn FnOnce(TaskContext) -> TaskResult + Send + 'static>;

#[derive(Clone)]
pub struct TaskContext {
    id: TaskId,
    cancelled: Arc<AtomicBool>,
}

impl TaskContext {
    pub(crate) fn new(id: TaskId, cancelled: Arc<AtomicBool>) -> Self {
        Self { id, cancelled }
    }

    pub fn id(&self) -> TaskId {
        self.id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check_cancelled(&self) -> Result<(), TaskError> {
        if self.is_cancelled() {
            Err(TaskError::Cancelled)
        } else {
            Ok(())
        }
    }
}

pub struct Task {
    pub id: TaskId,
    pub name: Option<String>,
    pub placement: Placement,
    state: AtomicU8,
    cancelled: Arc<AtomicBool>,
    body: Mutex<Option<TaskFn>>,
    completion: Arc<Completion>,
}

impl Task {
    pub fn new(
        name: Option<String>,
        placement: Placement,
        body: TaskFn,
    ) -> (Arc<Self>, TaskHandle) {
        let id = TaskId::next();
        let completion = Arc::new(Completion::default());
        let cancelled = Arc::new(AtomicBool::new(false));

        let task = Arc::new(Self {
            id,
            name,
            placement,
            state: AtomicU8::new(TaskState::Created as u8),
            cancelled,
            body: Mutex::new(Some(body)),
            completion: Arc::clone(&completion),
        });

        let handle = TaskHandle {
            id,
            task: Arc::clone(&task),
            completion,
        };

        (task, handle)
    }

    pub fn state(&self) -> TaskState {
        TaskState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub(crate) fn set_state(&self, state: TaskState) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        let state = self.state();
        if matches!(
            state,
            TaskState::Created | TaskState::Runnable | TaskState::Parked
        ) {
            self.set_state(TaskState::Cancelled);
            self.completion.finish(Err(TaskError::Cancelled));
        }
    }

    pub(crate) fn run(self: &Arc<Self>) {
        if self.cancelled.load(Ordering::Acquire) {
            self.set_state(TaskState::Cancelled);
            self.completion.finish(Err(TaskError::Cancelled));
            return;
        }

        self.set_state(TaskState::Running);
        let body = self
            .body
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();

        let Some(body) = body else {
            if !self.state().is_terminal() {
                self.set_state(TaskState::Failed);
                self.completion
                    .finish(Err(TaskError::Failed("task body already consumed".into())));
            }
            return;
        };

        let context = TaskContext::new(self.id, Arc::clone(&self.cancelled));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(context)));

        match result {
            Ok(Ok(value)) => {
                self.set_state(TaskState::Completed);
                self.completion.finish(Ok(value));
            }
            Ok(Err(error)) => {
                self.set_state(match error {
                    TaskError::Cancelled => TaskState::Cancelled,
                    _ => TaskState::Failed,
                });
                self.completion.finish(Err(error));
            }
            Err(_) => {
                self.set_state(TaskState::Failed);
                self.completion.finish(Err(TaskError::Panicked));
            }
        }
    }
}

#[derive(Default)]
struct Completion {
    result: Mutex<Option<TaskResult>>,
    ready: Condvar,
}

impl Completion {
    fn finish(&self, result: TaskResult) {
        let mut slot = self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if slot.is_none() {
            *slot = Some(result);
            self.ready.notify_all();
        }
    }
}

#[derive(Clone)]
pub struct TaskHandle {
    id: TaskId,
    task: Arc<Task>,
    completion: Arc<Completion>,
}

impl TaskHandle {
    pub fn id(&self) -> TaskId {
        self.id
    }

    pub fn state(&self) -> TaskState {
        self.task.state()
    }

    pub fn cancel(&self) {
        self.task.cancel();
    }

    pub fn join(self) -> TaskResult {
        let mut slot = self
            .completion
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        while slot.is_none() {
            slot = self
                .completion
                .ready
                .wait(slot)
                .unwrap_or_else(|poison| poison.into_inner());
        }

        slot.take().unwrap()
    }
}
