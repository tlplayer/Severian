//! Tensor lowering shared by the Linalg and StableHLO paths.
//!
//! `linalg` preserves Severian's existing CPU-oriented tensor kernels. Higher
//! level tensor lowering can use the type helpers here and select StableHLO when
//! the operation should be handed to XLA.

pub mod linalg;

pub(crate) use linalg::mlir_kernels;

use severian_hir::{TensorDimension, TensorElementType, TensorType, ValueType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLoweringTarget {
    Linalg,
    StableHlo,
}

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

pub fn choose_target(function: &str) -> TensorLoweringTarget {
    let name = function
        .rsplit_once('.')
        .map(|(_, leaf)| leaf)
        .unwrap_or(function)
        .to_ascii_lowercase();

    if [
        "matmul",
        "dot",
        "gemm",
        "conv",
        "attention",
        "softmax",
        "layer_norm",
        "layernorm",
        "transpose",
        "reshape",
        "broadcast",
        "reduce",
        "sum",
        "mean",
        "relu",
        "gelu",
    ]
    .iter()
    .any(|token| name.contains(token))
    {
        TensorLoweringTarget::StableHlo
    } else {
        TensorLoweringTarget::Linalg
    }
}
