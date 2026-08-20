use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    C,
    System,
    SysV64,
    Win64,
    Aapcs64,
    Rust,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Integer { bits: u16, signed: bool },
    Float { format: AbiFloatFormat },
    Boolean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbiFloatFormat {
    Ieee(u16),
    BrainFloat16,
}

impl AbiFloatFormat {
    pub const fn bits(self) -> u16 {
        match self {
            Self::Ieee(bits) => bits,
            Self::BrainFloat16 => 16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub ty: AbiType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordRepresentation {
    C,
    Packed(u32),
    Transparent,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordType {
    pub name: Option<String>,
    pub fields: Vec<Field>,
    pub representation: RecordRepresentation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumType {
    pub name: Option<String>,
    pub underlying: ScalarType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub convention: CallingConvention,
    pub parameters: Vec<AbiType>,
    pub result: Box<AbiType>,
    pub variadic: bool,
}

/// A concrete, target-layout-ready ABI type. Semantic language concepts do not
/// belong here; library-specific descriptors are ordinary records.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AbiType {
    Void,
    Scalar(ScalarType),
    Pointer {
        pointee: Box<AbiType>,
        mutable: bool,
    },
    Array {
        element: Box<AbiType>,
        length: u64,
    },
    Record(RecordType),
    Union(RecordType),
    Enum(EnumType),
    Function(Box<FunctionType>),
    Opaque {
        name: String,
    },
}

impl AbiType {
    pub const fn integer(bits: u16, signed: bool) -> Self {
        Self::Scalar(ScalarType::Integer { bits, signed })
    }

    pub const fn float(bits: u16) -> Self {
        Self::Scalar(ScalarType::Float {
            format: AbiFloatFormat::Ieee(bits),
        })
    }

    pub const fn bfloat16() -> Self {
        Self::Scalar(ScalarType::Float {
            format: AbiFloatFormat::BrainFloat16,
        })
    }

    pub fn pointer_to(pointee: Self, mutable: bool) -> Self {
        Self::Pointer {
            pointee: Box::new(pointee),
            mutable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolName(String);

impl SymbolName {
    pub fn new(value: impl Into<String>) -> Result<Self, SymbolError> {
        let value = value.into();
        if value.is_empty()
            || value
                .bytes()
                .any(|byte| byte == 0 || byte.is_ascii_whitespace())
        {
            return Err(SymbolError::InvalidName(value));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage {
    Internal,
    External,
    Weak,
    LinkOnce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    Default,
    Hidden,
    Protected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DllStorage {
    Default,
    Import,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Data,
    ThreadLocal,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub name: SymbolName,
    pub linkage: Linkage,
    pub visibility: Visibility,
    pub dll_storage: DllStorage,
    pub kind: SymbolKind,
}

impl Symbol {
    pub fn imported_function(name: impl Into<String>) -> Result<Self, SymbolError> {
        Ok(Self {
            name: SymbolName::new(name)?,
            linkage: Linkage::External,
            visibility: Visibility::Default,
            dll_storage: DllStorage::Import,
            kind: SymbolKind::Function,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolError {
    InvalidName(String),
}

impl fmt::Display for SymbolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid ABI symbol name: {self:?}")
    }
}

impl std::error::Error for SymbolError {}
