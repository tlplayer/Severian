use crate::AbiId;

/// Function calling convention. Unknown/custom conventions remain data rather
/// than becoming compiler enum variants for every runtime or vendor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    C,
    System,
    Named(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnwindPolicy {
    Forbidden,
    Allowed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbiSpec {
    pub id: AbiId,
    pub convention: CallingConvention,
    pub unwind: UnwindPolicy,
    pub supports_variadic: bool,
}

impl AbiSpec {
    pub fn new(id: AbiId, convention: CallingConvention) -> Self {
        Self {
            id,
            convention,
            unwind: UnwindPolicy::Forbidden,
            supports_variadic: false,
        }
    }

    pub fn c() -> Self {
        Self {
            id: AbiId::new("c"),
            convention: CallingConvention::C,
            unwind: UnwindPolicy::Forbidden,
            supports_variadic: true,
        }
    }

    pub fn system() -> Self {
        Self {
            id: AbiId::new("system"),
            convention: CallingConvention::System,
            unwind: UnwindPolicy::Forbidden,
            supports_variadic: true,
        }
    }
}
