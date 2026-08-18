use crate::{
    Capability, ExternalDeclaration, Implementation, ModuleId, ModuleInterface, PackageId,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Interface {
    pub id: PackageId,
    pub root: ModuleId,
    pub modules: Vec<ModuleInterface>,
    pub implementations: Vec<Implementation>,
    pub externals: Vec<ExternalDeclaration>,
    pub capabilities: Vec<Capability>,
}
