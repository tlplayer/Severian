//! StableHLO emission for XLA-bound tensor regions.
//!
//! Severian's lowering crate currently emits textual MLIR. This module follows
//! the same model and gives the compiler a small typed layer over StableHLO
//! instead of scattering operation syntax through the main lowering file.

pub mod activation;
pub mod attention;
pub mod convolution;
pub mod indexing;
pub mod linear;
pub mod normalization;
pub mod ops;
pub mod reduction;

pub use ops::{MlirValue, StableHloEmitter};
pub use reduction::StableHloReduction;

use severian_hir::{Expression, Function, Instruction, Program, ValueType};
use severian_hir::{TensorDimension, TensorElementType, TensorType};
use severian_mlir::Module;
use std::fmt::{self, Write};

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
    NoTensorFunctions,
    UnsupportedFunction {
        function: String,
        reason: String,
    },
}

impl fmt::Display for StableHloLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOperation(operation) => {
                write!(formatter, "unsupported StableHLO operation `{operation}`")
            }
            Self::InvalidArity {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "StableHLO operation `{operation}` expects {expected} arguments, got {actual}",
            ),
            Self::InvalidRank {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "StableHLO operation `{operation}` expects rank {expected}, got {actual:?}",
            ),
            Self::NoTensorFunctions => formatter
                .write_str("XLA lowering requires at least one function with a tensor result"),
            Self::UnsupportedFunction { function, reason } => write!(
                formatter,
                "cannot lower tensor function `{function}` to StableHLO: {reason}"
            ),
        }
    }
}

impl std::error::Error for StableHloLoweringError {}

/// Lowers the currently representable whole-program tensor boundary.
///
/// Typed HIR preserves tensor information and resolved call targets. This
/// initial whole-program path accepts tensor functions whose bodies directly
/// return one supported tensor operation over tensor parameters; composing
/// general tensor control flow belongs to the forthcoming MIR lowering.
pub fn lower_program(program: &Program) -> Result<Module, StableHloLoweringError> {
    let tensor_functions = program
        .functions
        .iter()
        .filter(|function| {
            function.native_symbol.is_none()
                && matches!(function.return_type, ValueType::Tensor(_))
        })
        .collect::<Vec<_>>();
    if tensor_functions.is_empty() {
        return Err(StableHloLoweringError::NoTensorFunctions);
    }

    let mut output = String::from("module {\n");
    for function in tensor_functions {
        lower_function(function, &mut output)?;
    }
    output.push_str("}\n");
    Ok(Module::new(output))
}

fn lower_function(function: &Function, output: &mut String) -> Result<(), StableHloLoweringError> {
    let ValueType::Tensor(result_type) = function.return_type else {
        unreachable!()
    };
    let mut arguments = Vec::with_capacity(function.params.len());
    for (index, parameter) in function.params.iter().enumerate() {
        let ValueType::Tensor(tensor) = parameter.ty else {
            return Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: format!("parameter `{}` is not a tensor", parameter.name),
            });
        };
        arguments.push((
            parameter.name.as_str(),
            argument(format!("%arg{index}"), tensor),
        ));
    }

    let Some(returned) = direct_tensor_return(&function.instructions) else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: function.name.clone(),
            reason: "expected a direct return of a tensor call".into(),
        });
    };
    let Expression::Call { target, args } = returned.kind() else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: function.name.clone(),
            reason: "expected a direct return of a tensor call".into(),
        });
    };

    let operands = args
        .iter()
        .map(|expression| match expression.kind() {
            Expression::Variable(name) => arguments
                .iter()
                .find(|(parameter, _)| *parameter == name.as_str())
                .map(|(_, value)| value.clone())
                .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                    function: function.name.clone(),
                    reason: format!(
                        "tensor operand `{name}` is not a parameter with a retained type"
                    ),
                }),
            _ => Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: "tensor call operands must currently be function parameters".into(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut emitter = StableHloEmitter::new();
    let result = lower_tensor_call(&target.name, &operands, result_type, &mut emitter)?;
    let signature = arguments
        .iter()
        .enumerate()
        .map(|(index, (_, value))| format!("%arg{index}: {}", value.ty))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "  func.func @{}({signature}) -> {} {{",
        function.name,
        tensor_type(result_type),
    )
    .expect("writing to a String cannot fail");
    output.push_str(emitter.as_str());
    writeln!(output, "    return {} : {}", result.name, result.ty)
        .expect("writing to a String cannot fail");
    output.push_str("  }\n");
    Ok(())
}

fn direct_tensor_return(instructions: &[Instruction]) -> Option<&Expression> {
    match instructions {
        [Instruction::Return(Some(value))] => Some(value),
        [Instruction::With { instructions, .. }] => direct_tensor_return(instructions),
        _ => None,
    }
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
        "add" | "rankedadd" | "tensor_add" => {
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

        "matmul" | "rankedmatmul" | "tensor_matmul" => {
            require_arity(&op, args, 2)?;
            match result_type.rank {
                Some(2) => Ok(linear::matmul_2d(emitter, &args[0], &args[1], result_type)),
                Some(3) => Ok(linear::linear_last_dimension(
                    emitter,
                    &args[0],
                    &args[1],
                    result_type,
                )),
                Some(4) => Ok(linear::batched_matmul(
                    emitter,
                    &args[0],
                    &args[1],
                    result_type,
                )),
                rank => Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: 2,
                    actual: rank,
                }),
            }
        }

        "reshape" | "tensor_reshape" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.reshape(&args[0], result_type))
        }

        "transpose" | "tensor_transpose" => {
            require_arity(&op, args, 1)?;
            require_rank(&op, result_type, 2)?;
            Ok(emitter.transpose(&args[0], &[1, 0], result_type))
        }

        "broadcast" | "broadcast_in_dim" | "tensor_broadcast" => {
            require_arity(&op, args, 1)?;
            let input_rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let result_rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            if input_rank > result_rank {
                return Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: usize::from(input_rank),
                    actual: Some(result_rank),
                });
            }
            let first = u64::from(result_rank - input_rank);
            let dimensions = (first..u64::from(result_rank)).collect::<Vec<_>>();
            Ok(emitter.broadcast_in_dim(&args[0], &dimensions, result_type))
        }

        "relu" | "rankedrelu" | "tensor_relu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::relu(emitter, &args[0], result_type))
        }

        "silu" | "swish" | "tensor_silu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::silu(emitter, &args[0], result_type))
        }

        "exp" | "exponential" | "tensor_exp" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.exponential(&args[0], result_type))
        }

        "tanh" | "tensor_tanh" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.tanh(&args[0], result_type))
        }

        "rsqrt" | "tensor_rsqrt" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.rsqrt(&args[0], result_type))
        }

        "sigmoid" | "logistic" | "tensor_sigmoid" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.logistic(&args[0], result_type))
        }

        "sum" | "reduce_sum" | "tensor_sum" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_sum(emitter, &args[0], &axes, result_type))
        }

        "max" | "reduce_max" | "tensor_max" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_max(emitter, &args[0], &axes, result_type))
        }

        "min" | "reduce_min" | "tensor_min" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_min(emitter, &args[0], &axes, result_type))
        }

        "mean" | "reduce_mean" | "tensor_mean" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            let count = static_reduction_count(&op, &args[0], &axes)?;
            Ok(reduction::mean(
                emitter,
                &args[0],
                &axes,
                result_type,
                count,
            ))
        }

        "gelu" | "tensor_gelu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::gelu_tanh(emitter, &args[0], result_type))
        }

        "softmax" | "softmax_last_axis" | "tensor_softmax" => {
            require_arity(&op, args, 1)?;
            let reduced_type = normalization::last_axis_reduced_type(result_type)?;
            Ok(normalization::softmax_last_axis(
                emitter,
                &args[0],
                result_type,
                reduced_type,
            ))
        }


        "rms_norm" | "rmsnorm" | "tensor_rms_norm" => {
            require_arity(&op, args, 2)?;
            let input_type = args[0]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let reduced_type = normalization::last_axis_reduced_type(input_type)?;
            let axis = u64::from(input_type.rank.unwrap() - 1);
            let hidden_size = static_reduction_count(&op, &args[0], &[axis])?;
            Ok(normalization::rms_norm(
                emitter,
                &args[0],
                &args[1],
                input_type,
                reduced_type,
                hidden_size,
                1e-5,
            ))
        }

        "layer_norm" | "layernorm" | "tensor_layer_norm" => {
            require_arity(&op, args, 3)?;
            let input_type = args[0]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let reduced_type = normalization::last_axis_reduced_type(input_type)?;
            let axis = u64::from(input_type.rank.unwrap() - 1);
            let hidden_size = static_reduction_count(&op, &args[0], &[axis])?;
            Ok(normalization::layer_norm(
                emitter,
                &args[0],
                &args[1],
                &args[2],
                input_type,
                reduced_type,
                hidden_size,
                1e-5,
            ))
        }

        // These are structurally represented in StableHLO but need shape or
        // axis metadata from semantic/HIR lowering before they can be emitted
        // without guessing. Keep them explicit instead of silently producing
        // wrong dimensions.
        "reduce" | "conv" | "convolution" | "attention" => {
            Err(StableHloLoweringError::UnsupportedOperation(op))
        }

        _ => Err(StableHloLoweringError::UnsupportedOperation(op)),
    }
}

fn reduced_suffix_axes(
    operation: &str,
    input: &MlirValue,
    result_type: TensorType,
) -> Result<Vec<u64>, StableHloLoweringError> {
    let input_rank = input
        .tensor_type()
        .and_then(|tensor| tensor.rank)
        .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "reduction input is missing ranked tensor metadata".into(),
        })?;
    let result_rank = result_type.rank.ok_or_else(|| {
        StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "reduction result is missing ranked tensor metadata".into(),
        }
    })?;
    if result_rank >= input_rank {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: format!(
                "suffix reduction must lower rank (input {input_rank}, result {result_rank})"
            ),
        });
    }
    Ok((u64::from(result_rank)..u64::from(input_rank)).collect())
}

fn static_reduction_count(
    operation: &str,
    input: &MlirValue,
    axes: &[u64],
) -> Result<u64, StableHloLoweringError> {
    let shape = input
        .ty
        .strip_prefix("tensor<")
        .and_then(|value| value.strip_suffix('>'))
        .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: format!("expected ranked tensor type, got {}", input.ty),
        })?;
    let dimensions = shape.split('x').collect::<Vec<_>>();
    axes.iter().try_fold(1u64, |count, &axis| {
        let dimension = dimensions
            .get(axis as usize)
            .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: format!("axis {axis} is outside type {}", input.ty),
            })?;
        let dimension = dimension.parse::<u64>().map_err(|_| {
            StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "mean/norm requires static reduced dimensions".into(),
            }
        })?;
        count.checked_mul(dimension).ok_or_else(|| {
            StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "reduction element count overflow".into(),
            }
        })
    })
}

pub fn argument(name: impl Into<String>, tensor: TensorType) -> MlirValue {
    MlirValue::from_tensor(name, tensor)
}

pub fn scalar_tensor(element: TensorElementType) -> TensorType {
    TensorType {
        element,
        rank: Some(0),
        dimensions: [TensorDimension::Dynamic; 8],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stablehlo::{
        attention::{llama_transformer_block, AttentionTypes, TransformerBlockTypes},
        normalization::{last_axis_reduced_type, softmax_last_axis},
    };

    fn tensor(rank: u8) -> TensorType {
        TensorType {
            element: TensorElementType::F32,
            rank: Some(rank),
            dimensions: [TensorDimension::Dynamic; 8],
        }
    }

    #[test]
    fn softmax_is_a_stablehlo_reduction_graph() {
        let input_type = tensor(4);
        let reduced_type = last_axis_reduced_type(input_type).unwrap();
        let input = argument("%input", input_type);
        let mut emitter = StableHloEmitter::new();
        softmax_last_axis(&mut emitter, &input, input_type, reduced_type);
        let text = emitter.as_str();
        assert!(text.contains("stablehlo.reduce"));
        assert!(text.contains("stablehlo.maximum"));
        assert!(text.contains("stablehlo.exponential"));
        assert!(text.contains("stablehlo.divide"));
        assert!(!text.contains("custom_call"));
    }

    #[test]
    fn llama_block_contains_attention_mlp_norms_and_residuals() {
        let model = tensor(3);
        let reduced_model = tensor(2);
        let weight = tensor(2);
        let norm_weight = tensor(1);
        let mask = argument("%mask", tensor(4));
        let input = argument("%input", model);
        let norm_a = argument("%norm_a", norm_weight);
        let norm_b = argument("%norm_b", norm_weight);
        let wq = argument("%wq", weight);
        let wk = argument("%wk", weight);
        let wv = argument("%wv", weight);
        let wo = argument("%wo", weight);
        let gate = argument("%gate", weight);
        let up = argument("%up", weight);
        let down = argument("%down", weight);
        let types = TransformerBlockTypes {
            model,
            reduced_model,
            attention: AttentionTypes {
                projected: tensor(3),
                projected_4d: tensor(4),
                qkv: tensor(4),
                key_transposed: tensor(4),
                scores: tensor(4),
                reduced_scores: tensor(3),
                context: tensor(4),
                context_transposed: tensor(4),
                merged_context: tensor(3),
                output: model,
            },
            mlp_intermediate: tensor(3),
        };
        let mut emitter = StableHloEmitter::new();
        llama_transformer_block(
            &mut emitter,
            &input,
            &norm_a,
            &wq,
            &wk,
            &wv,
            &wo,
            &mask,
            &norm_b,
            &gate,
            &up,
            &down,
            types,
            4096,
            128,
            1e-5,
        );
        let text = emitter.as_str();
        assert_eq!(text.matches("stablehlo.dot_general").count(), 9);
        assert!(text.matches("stablehlo.reduce").count() >= 4);
        assert!(text.contains("stablehlo.rsqrt"));
        assert!(text.contains("stablehlo.logistic"));
        assert!(!text.contains("custom_call"));
    }
}
