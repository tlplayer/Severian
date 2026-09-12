use std::time::Instant;

pub(crate) struct Stage {
    _hook: crate::hooks::Scope,
    name: &'static str,
    started: Option<Instant>,
}

impl Stage {
    #[track_caller]
    pub(crate) fn begin(name: &'static str) -> Self {
        Self {
            _hook: crate::hooks::Scope::enter(name),
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
