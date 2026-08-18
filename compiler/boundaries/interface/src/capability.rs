use crate::CapabilityId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capability {
    pub id: CapabilityId,
}

impl Capability {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: CapabilityId::new(id),
        }
    }
}
