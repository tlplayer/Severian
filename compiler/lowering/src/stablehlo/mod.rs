//! StableHLO emission for XLA-bound tensor regions.
//!
//! Severian's lowering crate currently emits textual MLIR. This module follows
//! the same model and gives the compiler a small typed layer over StableHLO
//! instead of scattering operation syntax through the main lowering file.

pub mod ops;

pub use ops::{MlirValue, StableHloEmitter};

use severian_hir::{TensorDimension, TensorElementType, TensorType};
use severian_hir::{Expression, Function, Instruction, Program, ValueType};
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
    UnsupportedFunction { function: String, reason: String },
}

impl fmt::Display for StableHloLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOperation(operation) =>
                write!(formatter, "unsupported StableHLO operation `{operation}`"),
            Self::InvalidArity { operation, expected, actual } => write!(
                formatter,
                "StableHLO operation `{operation}` expects {expected} arguments, got {actual}",
            ),
            Self::InvalidRank { operation, expected, actual } => write!(
                formatter,
                "StableHLO operation `{operation}` expects rank {expected}, got {actual:?}",
            ),
            Self::NoTensorFunctions => formatter.write_str(
                "XLA lowering requires at least one function with a tensor result",
            ),
            Self::UnsupportedFunction { function, reason } =>
                write!(formatter, "cannot lower tensor function `{function}` to StableHLO: {reason}"),
        }
    }
}

impl std::error::Error for StableHloLoweringError {}

/// Lowers the currently representable whole-program tensor boundary.
///
/// HIR preserves tensor element/rank/shape information on signatures, but not
/// on every expression. Consequently this accepts the unambiguous subset: a
/// tensor function whose body directly returns one supported tensor call over
/// tensor parameters. Other shapes fail explicitly until typed expression
/// results are retained in HIR.
pub fn lower_program(program: &Program) -> Result<Module, StableHloLoweringError> {
    let tensor_functions = program
        .functions
        .iter()
        .filter(|function| matches!(function.return_type, ValueType::Tensor(_)))
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

fn lower_function(
    function: &Function,
    output: &mut String,
) -> Result<(), StableHloLoweringError> {
    let ValueType::Tensor(result_type) = function.return_type else { unreachable!() };
    let mut arguments = Vec::with_capacity(function.params.len());
    for (index, parameter) in function.params.iter().enumerate() {
        let ValueType::Tensor(tensor) = parameter.ty else {
            return Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: format!("parameter `{}` is not a tensor", parameter.name),
            });
        };
        arguments.push((parameter.name.as_str(), argument(format!("%arg{index}"), tensor)));
    }

    let [Instruction::Return(Some(Expression::Call { function: operation, args }))] =
        function.instructions.as_slice()
    else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: function.name.clone(),
            reason: "HIR does not retain typed intermediate expressions; expected a direct return of a tensor call".into(),
        });
    };

    let operands = args
        .iter()
        .map(|expression| match expression {
            Expression::Variable(name) => arguments
                .iter()
                .find(|(parameter, _)| *parameter == name.as_str())
                .map(|(_, value)| value.clone())
                .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                    function: function.name.clone(),
                    reason: format!("tensor operand `{name}` is not a parameter with a retained type"),
                }),
            _ => Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: "tensor call operands need retained HIR expression types".into(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut emitter = StableHloEmitter::new();
    let result = lower_tensor_call(operation, &operands, result_type, &mut emitter)?;
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
    ).expect("writing to a String cannot fail");
    output.push_str(emitter.as_str());
    writeln!(output, "    return {} : {}", result.name, result.ty)
        .expect("writing to a String cannot fail");
    output.push_str("  }\n");
    Ok(())
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
