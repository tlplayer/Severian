use severian_backend::LoweredType;
use severian_hir::{TypeId, TypeKind, TypeTable};

pub fn lower(type_id: TypeId, types: &TypeTable) -> Result<LoweredType, String> {
    let TypeKind::Primitive(id) = types
        .get(type_id)
        .ok_or_else(|| format!("unknown TypeId {}", type_id.0))?;
    let definition = severian_primitives::definition(*id)
        .map_err(|error| format!("primitive lookup failed: {error:?}"))?;
    match definition.representation {
        "machine-signed" => Ok(LoweredType::I64),
        representation => Err(format!(
            "unsupported primitive representation `{representation}`"
        )),
    }
}
