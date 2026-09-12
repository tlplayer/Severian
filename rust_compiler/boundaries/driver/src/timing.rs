use std::time::Instant;

pub(crate) struct Stage {
    name: &'static str,
    started: Option<Instant>,
}

impl Stage {
    pub(crate) fn begin(name: &'static str) -> Self {
        Self {
            name,
            started: (std::env::var("SEVERIAN_PROFILE_ACTIVE").as_deref() == Ok("1"))
                .then(Instant::now),
        }
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            eprintln!(
                "  Stage {}: {:.6}s wall",
                self.name,
                started.elapsed().as_secs_f64()
            );
        }
    }
}
