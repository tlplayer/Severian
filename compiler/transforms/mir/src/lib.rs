#![forbid(unsafe_code)]

mod build;
#[path = "model/operation/mod.rs"]
mod operation;
#[path = "model/value/mod.rs"]
mod value;

pub use build::build;
pub use operation::Operation;
use severian_hir::BindingId;
pub use severian_hir::{CallType, FunctionId};
pub use value::{Value, ValueId};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub values: Vec<Value>,
    pub bindings: Vec<(BindingId, ValueId)>,
    pub globals: Vec<ValueId>,
    pub initializer: Block,
    pub functions: Vec<Function>,
    pub entry: Option<FunctionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub parameters: Vec<ValueId>,
    pub result: severian_universal::TypeId,
    pub body: Option<Block>,
    pub call_type: CallType,
}
