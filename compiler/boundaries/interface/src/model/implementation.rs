use crate::{Constraint, ExternalId, ImplementationId, InterfaceType, SymbolId, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct Implementation {
    pub id: ImplementationId,
    pub trait_id: Option<TypeId>,
    pub target: InterfaceType,
    pub methods: Vec<ImplementationMethod>,
    pub constraints: Vec<Constraint>,
    pub selector: Option<SymbolId>,
    pub source: ImplementationSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplementationMethod {
    pub requirement: SymbolId,
    pub implementation: SymbolId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImplementationSource {
    Severian,
    External(ExternalId),
    Generated,
}
