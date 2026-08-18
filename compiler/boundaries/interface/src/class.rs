use crate::{GenericParameter, InterfaceType, SymbolId, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct ClassInterface {
    pub type_id: TypeId,
    pub fields: Vec<FieldInterface>,
    pub methods: Vec<SymbolId>,
    pub traits: Vec<TypeId>,
    pub generics: Vec<GenericParameter>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldInterface {
    pub name: String,
    pub ty: InterfaceType,
    pub access: FieldAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldAccess {
    ReadWrite,
    ReadOnly,
    Hidden,
}
