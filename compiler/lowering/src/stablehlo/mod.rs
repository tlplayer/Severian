//! StableHLO emission for XLA-bound tensor regions.
//!
//! Severian's lowering crate currently emits textual MLIR. This module follows
//! the same model and gives the compiler a small typed layer over StableHLO
//! instead of scattering operation syntax through the main lowering file.

pub mod ops;

pub use ops::{MlirValue, StableHloEmitter};

use severian_hir::{TensorDimension, TensorElementType, TensorType};

use crate::tensor::tensor_type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StableHloLoweringError {
    UnsupportedOperation(String),
    InvalidArity {
        operation: String,
        expected: usize,
        actual: usize,
    },
    InvalidRank {
        operation: String,
        expected: usize,
        actual: Option<u8>,
    },
}

pub fn lower_tensor_call(
    function: &str,
    args: &[MlirValue],
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let op = function
        .rsplit_once('.')
        .map(|(_, leaf)| leaf)
        .unwrap_or(function)
        .to_ascii_lowercase();

    match op.as_str() {
        "add" | "tensor_add" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.add(&args[0], &args[1], result_type))
        }

        "sub" | "subtract" | "tensor_sub" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.subtract(&args[0], &args[1], result_type))
        }

        "mul" | "multiply" | "tensor_mul" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.multiply(&args[0], &args[1], result_type))
        }

        "div" | "divide" | "tensor_div" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.divide(&args[0], &args[1], result_type))
        }

        "matmul" | "matrix_multiply" | "tensor_matmul" => {
            require_arity(&op, args, 2)?;
            require_rank(&op, result_type, 2)?;
            Ok(emitter.dot_general(
                &args[0],
                &args[1],
                &[],
                &[],
                &[1],
                &[0],
                result_type,
            ))
        }

        "relu" | "tensor_relu" => {
            require_arity(&op, args, 1)?;
            let zero = scalar_zero(result_type, emitter);
            Ok(emitter.maximum(&args[0], &zero, result_type))
        }

        // These are structurally represented in StableHLO but need shape or
        // axis metadata from semantic/HIR lowering before they can be emitted
        // without guessing. Keep them explicit instead of silently producing
        // wrong dimensions.
        "reshape"
        | "transpose"
        | "broadcast"
        | "broadcast_in_dim"
        | "sum"
        | "reduce"
        | "mean"
        | "softmax"
        | "layer_norm"
        | "layernorm"
        | "conv"
        | "convolution"
        | "attention" => Err(StableHloLoweringError::UnsupportedOperation(op)),

        _ => Err(StableHloLoweringError::UnsupportedOperation(op)),
    }
}

pub fn argument(name: impl Into<String>, tensor: TensorType) -> MlirValue {
    MlirValue::new(name, tensor_type(tensor))
}

pub fn scalar_tensor(element: TensorElementType) -> TensorType {
    TensorType {
        element,
        rank: Some(0),
        dimensions: [TensorDimension::Dynamic; 8],
    }
}

fn scalar_zero(
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> MlirValue {
    let scalar = scalar_tensor(result_type.element);
    let zero = match result_type.element {
        TensorElementType::F32 | TensorElementType::F64 => "0.0",
        TensorElementType::I32 | TensorElementType::I64 => "0",
    };

    let scalar_value = emitter.constant_scalar(zero, scalar);

    match result_type.rank {
        Some(0) => scalar_value,
        Some(rank) => emitter.broadcast_in_dim(
            &scalar_value,
            &[],
            TensorType {
                rank: Some(rank),
                ..result_type
            },
        ),
        None => scalar_value,
    }
}

fn require_arity(
    operation: &str,
    args: &[MlirValue],
    expected: usize,
) -> Result<(), StableHloLoweringError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(StableHloLoweringError::InvalidArity {
            operation: operation.to_string(),
            expected,
            actual: args.len(),
        })
    }
}

fn require_rank(
    operation: &str,
    tensor: TensorType,
    expected: usize,
) -> Result<(), StableHloLoweringError> {
    if tensor.rank == Some(expected as u8) {
        Ok(())
    } else {
        Err(StableHloLoweringError::InvalidRank {
            operation: operation.to_string(),
            expected,
            actual: tensor.rank,
        })
    }
}
