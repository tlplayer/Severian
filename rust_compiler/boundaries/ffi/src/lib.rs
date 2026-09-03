#![forbid(unsafe_code)]

mod convert;
mod model;
mod validate;

pub use convert::{lower_function, BoundaryPlan, Conversion, LoweredParameter};
pub use model::{
    AbiSelection, ForeignFunction, ForeignModule, ForeignParameter, ForeignTypeDeclaration,
    ForeignTypeRef, Lifetime, Ownership, ParameterMode, ValueContract,
};
pub use validate::{validate_function, FfiError};
