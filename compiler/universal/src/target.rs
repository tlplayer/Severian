#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLayout {
    pub pointer_bits: u16,
    pub machine_integer_bits: u16,
    pub machine_float_bits: u16,
}

impl TargetLayout {
    pub fn host() -> Self {
        Self {
            pointer_bits: usize::BITS as u16,
            machine_integer_bits: 64,
            machine_float_bits: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSpec {
    pub name: String,
    pub layout: TargetLayout,
}

impl TargetSpec {
    pub fn host() -> Self {
        Self {
            name: std::env::consts::ARCH.to_owned(),
            layout: TargetLayout::host(),
        }
    }
}
