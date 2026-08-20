use crate::{Constraint, GenericParameter, InterfaceType, SymbolId, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct TraitInterface {
    pub type_id: TypeId,
    pub methods: Vec<SymbolId>,
    pub associated_types: Vec<AssociatedType>,
    pub generics: Vec<GenericParameter>,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssociatedType {
    pub name: String,
    pub default: Option<InterfaceType>,
    pub constraints: Vec<Constraint>,
}
