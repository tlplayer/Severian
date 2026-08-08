use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct Deadline(Instant);

impl Deadline {
    pub fn after(duration: Duration) -> Self {
        Self(Instant::now() + duration)
    }

    pub fn at(instant: Instant) -> Self {
        Self(instant)
    }

    pub fn expired(self) -> bool {
        Instant::now() >= self.0
    }

    pub fn remaining(self) -> Duration {
        self.0.saturating_duration_since(Instant::now())
    }

    pub fn instant(self) -> Instant {
        self.0
    }
}
