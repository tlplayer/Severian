use crate::{
    Capability, CompileType, ExternalDeclaration, Implementation, ModuleId, ModuleInterface,
    PackageId,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Interface {
    pub id: PackageId,
    pub root: ModuleId,
    pub modules: Vec<ModuleInterface>,

    /// Compiler domains owned by this package.
    pub compile_types: Vec<CompileType>,

    pub implementations: Vec<Implementation>,
    pub externals: Vec<ExternalDeclaration>,
    pub capabilities: Vec<Capability>,
}
