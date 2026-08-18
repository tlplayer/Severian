use severian_abi::AbiId;

use crate::{ExternalId, ProviderId, SymbolId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalDeclaration {
    pub id: ExternalId,
    pub symbol: SymbolId,
    pub external_name: String,
    pub abi: AbiId,
    pub provider: ProviderId,
}
