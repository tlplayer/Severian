use std::sync::{Arc, Condvar, Mutex};

#[derive(Clone)]
pub struct ParkHandle {
    inner: Arc<(Mutex<bool>, Condvar)>,
}

pub struct Parker {
    handle: ParkHandle,
}

impl Parker {
    pub fn new() -> Self {
        Self {
            handle: ParkHandle {
                inner: Arc::new((Mutex::new(false), Condvar::new())),
            },
        }
    }

    pub fn handle(&self) -> ParkHandle {
        self.handle.clone()
    }

    pub fn park(&self) {
        let (flag, cv) = &*self.handle.inner;
        let mut wake = flag.lock().unwrap_or_else(|p| p.into_inner());

        while !*wake {
            wake = cv.wait(wake).unwrap_or_else(|p| p.into_inner());
        }

        *wake = false;
    }
}

impl ParkHandle {
    pub fn unpark(&self) {
        let (flag, cv) = &*self.inner;
        let mut wake = flag.lock().unwrap_or_else(|p| p.into_inner());
        *wake = true;
        cv.notify_one();
    }
}

impl Default for Parker {
    fn default() -> Self {
        Self::new()
    }
}
