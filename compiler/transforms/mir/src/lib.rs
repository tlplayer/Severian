#![forbid(unsafe_code)]

mod build;
#[path = "model/operation/mod.rs"]
mod operation;
#[path = "model/value/mod.rs"]
mod value;

pub use build::build;
pub use operation::Operation;
use severian_hir::BindingId;
pub use value::{Value, ValueId};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub operations: Vec<Operation>,
    pub bindings: Vec<(BindingId, ValueId)>,
}
