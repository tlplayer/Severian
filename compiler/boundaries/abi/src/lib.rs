#![forbid(unsafe_code)]

mod classify;
mod layout;
mod model;
mod target;

pub use classify::{
    classify_signature, AbiSignature, ClassifiedSignature, ClassifiedValue, PassMode, RegisterClass,
};
pub use layout::{align_to, layout_of, FieldLayout, Layout, LayoutError, LayoutKind};
pub use model::{
    AbiFloatFormat, AbiType, CallingConvention, DllStorage, EnumType, Field, FunctionType, Linkage,
    RecordRepresentation, RecordType, ScalarType, Symbol, SymbolKind, SymbolName, Visibility,
};
pub use target::{
    AbiTarget, Endianness, ScalarLayout, TargetDataLayout,
};
