use crate::{CompileTypeId, GenericParameter, InterfaceType, SymbolId, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct ClassInterface {
    pub type_id: TypeId,
    pub fields: Vec<FieldInterface>,
    pub methods: Vec<SymbolId>,
    pub traits: Vec<TypeId>,
    pub generics: Vec<GenericParameter>,

    /// Compiler domain for this class. `None` means ordinary core compilation.
    pub compile_type: Option<CompileTypeId>,
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
