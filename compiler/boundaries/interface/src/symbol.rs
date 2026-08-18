use crate::{
    ClassInterface, ConstantValue, EnumInterface, FunctionInterface, InterfaceType, SymbolId,
    TraitInterface,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub visibility: Visibility,
    pub kind: SymbolKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Package,
    Hidden,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SymbolKind {
    Function(FunctionInterface),
    Class(ClassInterface),
    Trait(TraitInterface),
    Enum(EnumInterface),
    Constant(ConstantInterface),
    TypeAlias(TypeAliasInterface),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantInterface {
    pub ty: InterfaceType,
    pub value: ConstantValue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeAliasInterface {
    pub target: InterfaceType,
}
