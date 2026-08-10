use super::compile::RawLoadedExecutable;

pub struct LoadedExecutable {
    _raw: RawLoadedExecutable,
}

impl LoadedExecutable {
    pub(crate) fn from_raw(raw: RawLoadedExecutable) -> Self { Self { _raw: raw } }
}
