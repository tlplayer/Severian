use crate::{CompileTypeId, SymbolId};

/// A compiler domain owned by a package.
///
/// The core compiler does not interpret the domain. Once MIR has reduced an
/// operation and tagged it with this id, routing resolves `handler` and gives
/// the reduced operation to that package. The handler returns MLIR to the core
/// compiler pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileType {
    pub id: CompileTypeId,

    /// Human-readable identity used for diagnostics and interface inspection.
    /// Routing always uses `id`, never this string.
    pub name: String,

    /// Package-owned compiler entrypoint for reduced operations in this domain.
    pub handler: SymbolId,
}
