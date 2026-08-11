use super::{
    buffer::Buffer,
    compile::{RawClient, RawLoadedExecutable},
    device::Device,
};
use crate::Result;
use std::sync::Arc;

pub struct LoadedExecutable {
    raw: RawLoadedExecutable,
    client: Arc<RawClient>,
}

impl LoadedExecutable {
    pub(crate) fn from_raw(raw: RawLoadedExecutable, client: Arc<RawClient>) -> Self {
        Self { raw, client }
    }

    pub fn execute(&self, arguments: &[&Buffer], device: &Device) -> Result<Vec<Buffer>> {
        super::execute::execute(&self.raw, Arc::clone(&self.client), arguments, device)
    }
}
