use super::ParkHandle;

#[derive(Clone)]
pub struct WakeHandle {
    parker: ParkHandle,
}

impl WakeHandle {
    pub fn new(parker: ParkHandle) -> Self {
        Self { parker }
    }

    pub fn wake(&self) {
        self.parker.unpark();
    }
}
