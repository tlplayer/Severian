use std::collections::HashMap;

use crate::{
    instantiate_schema, validate_schema_parameters, validate_signature, AbiArgument, AbiError, AbiId,
    AbiInstance, AbiSchema, AbiSchemaId, AbiSpec, AbiSignature, InstantiateError,
};

#[derive(Clone, Debug, Default)]
pub struct AbiRegistry {
    specs: HashMap<AbiId, AbiSpec>,
    schemas: HashMap<AbiSchemaId, AbiSchema>,
}

impl AbiRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_builtin_abis() -> Self {
        let mut registry = Self::new();
        registry.insert_spec_unchecked(AbiSpec::c());
        registry.insert_spec_unchecked(AbiSpec::system());
        registry
    }

    pub fn register_spec(&mut self, spec: AbiSpec) -> Result<(), AbiError> {
        if self.specs.contains_key(&spec.id) {
            return Err(AbiError::DuplicateAbi(spec.id));
        }
        self.insert_spec_unchecked(spec);
        Ok(())
    }

    pub fn spec(&self, id: &AbiId) -> Option<&AbiSpec> {
        self.specs.get(id)
    }

    pub fn require_spec(&self, id: &AbiId) -> Result<&AbiSpec, AbiError> {
        self.spec(id).ok_or_else(|| AbiError::UnknownAbi(id.clone()))
    }

    pub fn validate(&self, signature: &AbiSignature) -> Result<(), AbiError> {
        let spec = self.require_spec(&signature.abi)?;
        validate_signature(signature, spec)
    }

    pub fn register_schema(&mut self, schema: AbiSchema) -> Result<(), RegisterSchemaError> {
        validate_schema_parameters(&schema).map_err(RegisterSchemaError::Invalid)?;
        if self.schemas.contains_key(&schema.id) {
            return Err(RegisterSchemaError::Abi(AbiError::DuplicateSchema(schema.id)));
        }
        self.schemas.insert(schema.id.clone(), schema);
        Ok(())
    }

    pub fn schema(&self, id: &AbiSchemaId) -> Option<&AbiSchema> {
        self.schemas.get(id)
    }

    pub fn require_schema(&self, id: &AbiSchemaId) -> Result<&AbiSchema, AbiError> {
        self.schema(id).ok_or_else(|| AbiError::UnknownSchema(id.clone()))
    }

    pub fn instantiate(
        &self,
        schema: &AbiSchemaId,
        arguments: Vec<AbiArgument>,
    ) -> Result<AbiInstance, InstantiateError> {
        instantiate_schema(self, schema, arguments)
    }

    pub fn specs(&self) -> impl Iterator<Item = &AbiSpec> {
        self.specs.values()
    }

    pub fn schemas(&self) -> impl Iterator<Item = &AbiSchema> {
        self.schemas.values()
    }

    fn insert_spec_unchecked(&mut self, spec: AbiSpec) {
        self.specs.insert(spec.id.clone(), spec);
    }
}

#[derive(Debug)]
pub enum RegisterSchemaError {
    Abi(AbiError),
    Invalid(InstantiateError),
}

impl std::fmt::Display for RegisterSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Abi(e) => e.fmt(f),
            Self::Invalid(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for RegisterSchemaError {}
