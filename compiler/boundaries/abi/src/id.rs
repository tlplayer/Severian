use std::fmt;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

string_id!(AbiId);
string_id!(AbiSchemaId);
string_id!(AddressSpaceId);
string_id!(OpaqueId);
string_id!(RecordId);
string_id!(ResourceId);
string_id!(UnionId);

impl AddressSpaceId {
    /// The target's ordinary/default pointer address space.
    pub fn default_space() -> Self {
        Self::new("default")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SchemaParamId(pub u16);

impl SchemaParamId {
    pub const fn new(index: u16) -> Self {
        Self(index)
    }
}
