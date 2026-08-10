use super::compile::RawLoadedExecutable;

pub struct LoadedExecutable {
    raw: RawLoadedExecutable,
}

impl LoadedExecutable {
    pub(crate) fn from_raw(raw: RawLoadedExecutable) -> Self { Self { raw } }

    pub(crate) fn raw(&self) -> &RawLoadedExecutable { &self.raw }
}
