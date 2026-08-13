#![forbid(unsafe_code)]

mod builder;
mod ir;
mod verify;

pub use builder::lower;
pub use ir::*;
pub use verify::*;
