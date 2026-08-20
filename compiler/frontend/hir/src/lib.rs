#![forbid(unsafe_code)]

#[path = "model/expression/mod.rs"]
mod expression;
#[path = "model/statement/mod.rs"]
mod statement;

pub use expression::{Expression, ExpressionKind};
use severian_interface::PrimitiveId;
pub use statement::Binding;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HirId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Primitive(PrimitiveId),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeTable {
    types: Vec<TypeKind>,
}

impl TypeTable {
    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if let Some(index) = self.types.iter().position(|known| known == &kind) {
            return TypeId(index as u32);
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(kind);
        id
    }
    pub fn get(&self, id: TypeId) -> Option<&TypeKind> {
        self.types.get(id.0 as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Module {
    pub bindings: Vec<Binding>,
    pub types: TypeTable,
}
