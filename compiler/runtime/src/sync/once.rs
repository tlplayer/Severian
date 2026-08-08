use std::sync::OnceLock;

pub struct SeverianOnce<T> {
    inner: OnceLock<T>,
}

impl<T> SeverianOnce<T> {
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Option<&T> {
        self.inner.get()
    }

    pub fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        self.inner.get_or_init(init)
    }

    pub fn set(&self, value: T) -> Result<(), T> {
        self.inner.set(value)
    }
}

impl<T> Default for SeverianOnce<T> {
    fn default() -> Self {
        Self::new()
    }
}
