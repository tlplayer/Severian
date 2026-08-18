use crate::{ConstantValue, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct EnumInterface {
    pub type_id: TypeId,
    pub variants: Vec<EnumVariant>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub value: Option<ConstantValue>,
}
