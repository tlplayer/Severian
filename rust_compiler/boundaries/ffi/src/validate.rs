use crate::{ForeignFunction, ForeignModule, ForeignTypeRef, Lifetime, Ownership, ParameterMode};
use severian_abi::{layout_of, AbiTarget, AbiType};
use std::collections::BTreeSet;
use std::fmt;

pub fn validate_function(
    function: &ForeignFunction,
    module: &ForeignModule,
    target: &AbiTarget,
) -> Result<(), FfiError> {
    let mut names = BTreeSet::new();
    for parameter in &function.parameters {
        if !names.insert(parameter.name.as_str()) {
            return Err(FfiError::DuplicateParameter(parameter.name.clone()));
        }
        validate_type_ref(&parameter.contract.ty, module)?;
        match (&parameter.mode, &parameter.contract.ownership) {
            (ParameterMode::Out | ParameterMode::InOut, Ownership::Copy) => {
                return Err(FfiError::OutputMustCrossByReference(parameter.name.clone()))
            }
            (ParameterMode::Out, Ownership::Borrowed(_)) => {
                return Err(FfiError::BorrowedOutput(parameter.name.clone()))
            }
            _ => {}
        }
    }
    validate_type_ref(&function.result.ty, module)?;
    if matches!(
        function.result.ownership,
        Ownership::Borrowed(Lifetime::Call)
    ) {
        return Err(FfiError::ReturnCannotBorrowCall);
    }
    for declaration in &module.types {
        if !matches!(declaration.representation, AbiType::Opaque { .. }) {
            layout_of(&declaration.representation, &target.data_layout)
                .map_err(|error| FfiError::Abi(error.to_string()))?;
        }
    }
    Ok(())
}

fn validate_type_ref(ty: &ForeignTypeRef, module: &ForeignModule) -> Result<(), FfiError> {
    match ty {
        ForeignTypeRef::Severian(_) => Ok(()),
        ForeignTypeRef::External(name) if module.type_declaration(name).is_some() => Ok(()),
        ForeignTypeRef::External(name) => Err(FfiError::UnknownExternalType(name.clone())),
        ForeignTypeRef::Pointer { pointee, .. } => validate_type_ref(pointee, module),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiError {
    Abi(String),
    BorrowedOutput(String),
    DuplicateParameter(String),
    InvalidOwnership(String),
    NotPrimitive(severian_universal::TypeId),
    OutputMustCrossByReference(String),
    ReturnCannotBorrowCall,
    UnknownExternalType(String),
    UnsupportedRepresentation(String),
}

impl fmt::Display for FfiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid foreign interface: {self:?}")
    }
}

impl std::error::Error for FfiError {}
