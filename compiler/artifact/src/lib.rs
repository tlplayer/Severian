#![forbid(unsafe_code)]

/// Identity assigned by the planner to one typed custom MIR region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompiledRegionId(u32);

impl CompiledRegionId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Identity assigned centrally to the artifact produced for a compiled region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactId(u32);

impl ArtifactId {
    pub const fn for_region(region: CompiledRegionId) -> Self {
        Self(region.index())
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}
