#![forbid(unsafe_code)]

#[path = "core/types/primitive/mod.rs"]
mod primitive;

use severian_backend::{LoweredModule, Operation as LoweredOperation, ValueId as LoweredValueId};
use severian_hir::TypeTable;
use severian_mir::{Module as MirModule, Operation as MirOperation};

pub fn lower(mir: &MirModule, types: &TypeTable) -> Result<LoweredModule, String> {
    let values = mir
        .values
        .iter()
        .map(|value| primitive::lower(value.type_id, types))
        .collect::<Result<Vec<_>, _>>()?;
    let operations = mir
        .operations
        .iter()
        .map(|operation| match operation {
            MirOperation::ConstantInt { value, result } => LoweredOperation::ConstantI64 {
                value: *value,
                result: LoweredValueId(result.0),
            },
            MirOperation::AddInt {
                left,
                right,
                result,
            } => LoweredOperation::AddI64 {
                left: LoweredValueId(left.0),
                right: LoweredValueId(right.0),
                result: LoweredValueId(result.0),
            },
        })
        .collect();
    Ok(LoweredModule {
        values,
        operations,
        last_binding: mir
            .bindings
            .last()
            .map(|(_, value)| LoweredValueId(value.0)),
    })
}
