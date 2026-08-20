#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeclarationId(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveId(pub DeclarationId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveInterface {
    pub id: PrimitiveId,
    pub path: &'static str,
    pub category: &'static str,
    pub representation: &'static str,
    pub signed: bool,
    pub default_literal: bool,
}
