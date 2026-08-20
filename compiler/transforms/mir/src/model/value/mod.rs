use severian_universal::TypeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value {
    pub id: ValueId,
    pub type_id: TypeId,
}
