#![forbid(unsafe_code)]

mod builder;
mod error;
mod ir;
mod verify;

pub use builder::lower;
pub use error::MirLoweringError;
pub use ir::*;
pub use tensor::*;
pub use verify::*;
