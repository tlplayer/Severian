#![forbid(unsafe_code)]

use severian_interface::{DeclarationId, PrimitiveId, PrimitiveInterface};

const INTEGER_CONTRACT: &str = include_str!("integers.sev");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapError {
    MissingIntegerDeclaration,
    InvalidIntegerContract(&'static str),
}

pub fn load() -> Result<Vec<PrimitiveInterface>, BootstrapError> {
    let declaration = "trait int: Primitive + Integer[int]:";
    let start = INTEGER_CONTRACT
        .find(declaration)
        .ok_or(BootstrapError::MissingIntegerDeclaration)?;
    let tail = &INTEGER_CONTRACT[start..];
    let end = tail.find("\n\n\ntrait ").unwrap_or(tail.len());
    let contract = &tail[..end];
    for required in [
        "property category: string = \"integer\"",
        "property representation: string = \"machine-signed\"",
        "property signed: bool = true",
        "property default_literal: bool = true",
    ] {
        if !contract.contains(required) {
            return Err(BootstrapError::InvalidIntegerContract(required));
        }
    }
    Ok(vec![PrimitiveInterface {
        id: PrimitiveId(DeclarationId(1)),
        path: "core.primitives.int",
        category: "integer",
        representation: "machine-signed",
        signed: true,
        default_literal: true,
    }])
}

pub fn default_integer() -> Result<PrimitiveInterface, BootstrapError> {
    load()?
        .into_iter()
        .find(|primitive| primitive.category == "integer" && primitive.default_literal)
        .ok_or(BootstrapError::MissingIntegerDeclaration)
}

pub fn definition(id: PrimitiveId) -> Result<PrimitiveInterface, BootstrapError> {
    load()?
        .into_iter()
        .find(|primitive| primitive.id == id)
        .ok_or(BootstrapError::MissingIntegerDeclaration)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_identity_comes_from_bootstrap_contract() {
        let integer = default_integer().unwrap();
        assert_eq!(integer.path, "core.primitives.int");
        assert_eq!(integer.representation, "machine-signed");
    }
}
