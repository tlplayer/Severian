use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PackageId {
    pub name: String,
    pub version: String,
}

impl PackageId {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl fmt::Display for PackageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId {
    pub package: PackageId,
    pub local: u32,
}

impl ModuleId {
    pub fn new(package: PackageId, local: u32) -> Self {
        Self { package, local }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId {
    pub module: ModuleId,
    pub local: u32,
}

impl SymbolId {
    pub fn new(module: ModuleId, local: u32) -> Self {
        Self { module, local }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub SymbolId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenericId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplementationId {
    pub package: PackageId,
    pub local: u32,
}

impl ImplementationId {
    pub fn new(package: PackageId, local: u32) -> Self {
        Self { package, local }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExternalId {
    pub package: PackageId,
    pub local: u32,
}

impl ExternalId {
    pub fn new(package: PackageId, local: u32) -> Self {
        Self { package, local }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CompileTypeId {
    pub package: PackageId,
    pub local: u32,
}

impl CompileTypeId {
    pub fn new(package: PackageId, local: u32) -> Self {
        Self { package, local }
    }
}

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

string_id!(CapabilityId);
string_id!(ProviderId);
string_id!(IntrinsicId);
