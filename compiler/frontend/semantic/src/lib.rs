#![forbid(unsafe_code)]

mod analyzer;
mod native_abi;
pub use analyzer::{
    analyze, analyze_with_interfaces, analyze_with_packages, attach_module_metadata,
    attach_module_metadata_to, attach_module_metadata_to_with_packages,
    attach_module_metadata_with_packages, enforce_type_resolution_policy, SemanticError,
};
pub use native_abi::validate_native_abi;
