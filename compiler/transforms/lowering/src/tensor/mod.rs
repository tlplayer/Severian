//! Backend-independent tensor type formatting plus the Linalg emitter.
//!
//! Backend policy operates on resolved MIR tensor operations in `kernel`; this
//! module does not infer a backend from source function names.

pub mod linalg;

pub(crate) use linalg::mlir_kernels;

use severian_hir::{TensorDimension, TensorElementType, TensorType, ValueType};

pub fn element_type(element: TensorElementType) -> &'static str {
    element.mlir_name()
}

pub fn tensor_type(ty: TensorType) -> String {
    let element = element_type(ty.element);

    let Some(rank) = ty.rank else {
        return format!("tensor<*x{element}>");
    };

    if rank == 0 {
        return format!("tensor<{element}>");
    }

    let dimensions = (0..rank as usize)
        .map(|axis| match ty.dimensions[axis] {
            TensorDimension::Static(size) => size.to_string(),
            TensorDimension::Dynamic => "?".to_string(),
        })
        .collect::<Vec<_>>()
        .join("x");

    format!("tensor<{dimensions}x{element}>")
}

pub fn memref_type(ty: TensorType) -> String {
    let element = element_type(ty.element);

    let Some(rank) = ty.rank else {
        return format!("memref<*x{element}>");
    };

    if rank == 0 {
        return format!("memref<{element}>");
    }

    let dimensions = (0..rank as usize)
        .map(|axis| match ty.dimensions[axis] {
            TensorDimension::Static(size) => size.to_string(),
            TensorDimension::Dynamic => "?".to_string(),
        })
        .collect::<Vec<_>>()
        .join("x");

    format!("memref<{dimensions}x{element}>")
}

pub fn ranked_shape(ty: TensorType) -> Option<Vec<Option<u64>>> {
    let rank = ty.rank? as usize;
    Some(
        (0..rank)
            .map(|axis| match ty.dimensions[axis] {
                TensorDimension::Static(size) => Some(size),
                TensorDimension::Dynamic => None,
            })
            .collect(),
    )
}

pub fn as_tensor_type(ty: ValueType) -> Option<TensorType> {
    match ty {
        ValueType::Tensor(tensor) => Some(tensor),
        _ => None,
    }
}
