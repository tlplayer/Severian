#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeType {
    I64,
}

impl NativeType {
    pub const fn c_spelling(self) -> &'static str {
        match self {
            Self::I64 => "int64_t",
        }
    }
}
