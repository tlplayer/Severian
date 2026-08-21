use std::fmt;

/// Stable declaration identity derived from its canonical package path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeclarationId(pub u128);

impl DeclarationId {
    pub fn from_path(path: &str) -> Self {
        // FNV-1a is deliberately specified here rather than using RandomState or
        // DefaultHasher, whose output is not a persistent format contract.
        const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        let mut hash = OFFSET;
        for byte in path.as_bytes() {
            hash ^= u128::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        Self(hash)
    }
}

impl fmt::Display for DeclarationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveId(pub DeclarationId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompilerId(DeclarationId);

impl CompilerId {
    pub(crate) const fn from_declaration(declaration: DeclarationId) -> Self {
        Self(declaration)
    }

    pub const fn declaration(self) -> DeclarationId {
        self.0
    }
}

impl fmt::Display for CompilerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_depends_on_path_not_registration_order() {
        let before = DeclarationId::from_path("core.primitives.i32");
        let unrelated = DeclarationId::from_path("core.primitives.f128");
        let after = DeclarationId::from_path("core.primitives.i32");
        assert_eq!(before, after);
        assert_ne!(before, unrelated);
    }
}
