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

use severian_hir::{
    BindingId, CallTarget, Expression, Function, FunctionId, Instruction, Program, ValueType,
};
use severian_hir::{TensorDimension, TensorElementType, TensorType};
use severian_mlir::Module;
use std::collections::HashMap;
use std::fmt::{self, Write};

use crate::tensor::tensor_type;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StableHloLoweringError {
    UnknownFunction(FunctionId),
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
            Self::UnknownFunction(id) => {
                write!(formatter, "unknown StableHLO entry function {id:?}")
            }
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
/// whole-program path accepts straight-line tensor SSA, including nested calls,
/// local bindings, reassignment, and placement regions. Control flow remains a
/// MIR concern, but model kernels no longer have to be single-expression shims.
pub fn lower_program(program: &Program) -> Result<Module, StableHloLoweringError> {
    let tensor_functions = program
        .functions
        .iter()
        .filter(|function| {
            function.native_symbol.is_none()
                && matches!(function.return_type, ValueType::Tensor(_))
                && function
                    .decorators
                    .iter()
                    .any(|decorator| decorator.package == "tensor")
        })
        .collect::<Vec<_>>();
    if tensor_functions.is_empty() {
        return Err(StableHloLoweringError::NoTensorFunctions);
    }

    let context = LoweringContext::new(program);
    let mut output = String::from("module {\n");
    for function in tensor_functions {
        lower_function(&context, function, &function.name, &mut output)?;
    }
    output.push_str("}\n");
    Ok(Module::new(output))
}

/// Lowers one resolved Severian function as a single StableHLO/XLA region.
/// Calls to other source functions are inlined by `FunctionId`; native tensor
/// functions remain generic StableHLO intrinsics.
pub fn lower_entry(program: &Program, entry: FunctionId) -> Result<Module, StableHloLoweringError> {
    let context = LoweringContext::new(program);
    let function = context
        .functions
        .get(&entry)
        .copied()
        .ok_or(StableHloLoweringError::UnknownFunction(entry))?;
    let mut output = String::from("module {\n");
    lower_function(&context, function, "main", &mut output)?;
    output.push_str("}\n");
    Ok(Module::new(output))
}

struct LoweringContext<'a> {
    functions: HashMap<FunctionId, &'a Function>,
}

impl<'a> LoweringContext<'a> {
    fn new(program: &'a Program) -> Self {
        let functions = program
            .functions
            .iter()
            .map(|function| (function.id, function))
            .collect();
        Self { functions }
    }
}

fn lower_function(
    context: &LoweringContext<'_>,
    function: &Function,
    exported_name: &str,
    output: &mut String,
) -> Result<(), StableHloLoweringError> {
    let ValueType::Tensor(result_type) = function.return_type else {
        unreachable!()
    };
    let mut arguments = Vec::with_capacity(function.params.len());
    let mut values = HashMap::new();
    for (index, parameter) in function.params.iter().enumerate() {
        let ValueType::Tensor(tensor) = parameter.ty else {
            return Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: format!("parameter `{}` is not a tensor", parameter.name),
            });
        };
        let value = argument(format!("%arg{index}"), tensor);
        arguments.push(value.clone());
        values.insert(parameter.name.id, value);
    }
    let mut emitter = StableHloEmitter::new();
    let result = lower_straight_line(
        context,
        &function.name,
        result_type,
        &function.instructions,
        &mut values,
        &mut emitter,
    )?
    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
        function: function.name.clone(),
        reason: "tensor function has no returned value".into(),
    })?;
    let signature = arguments
        .iter()
        .enumerate()
        .map(|(index, value)| format!("%arg{index}: {}", value.ty))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "  func.func @{}({signature}) -> {} {{",
        exported_name,
        tensor_type(result_type),
    )
    .expect("writing to a String cannot fail");
    output.push_str(emitter.as_str());
    writeln!(output, "    return {} : {}", result.name, result.ty)
        .expect("writing to a String cannot fail");
    output.push_str("  }\n");
    Ok(())
}

fn lower_straight_line(
    context: &LoweringContext<'_>,
    function: &str,
    return_type: TensorType,
    instructions: &[Instruction],
    values: &mut HashMap<BindingId, MlirValue>,
    emitter: &mut StableHloEmitter,
) -> Result<Option<MlirValue>, StableHloLoweringError> {
    for instruction in instructions {
        match instruction {
            Instruction::Let { name, value } | Instruction::TryLet { name, value, .. } => {
                let value = lower_expression(context, function, value, None, values, emitter)?;
                values.insert(name.id, value);
            }
            Instruction::Assign { target, op, value } => {
                if !matches!(op, severian_hir::AssignmentOp::Assign) {
                    return Err(StableHloLoweringError::UnsupportedFunction {
                        function: function.into(),
                        reason: "compound tensor assignment must be normalized before StableHLO"
                            .into(),
                    });
                }
                let Expression::Variable(name) = target.kind() else {
                    return Err(StableHloLoweringError::UnsupportedFunction {
                        function: function.into(),
                        reason: "tensor assignment target must be a local variable".into(),
                    });
                };
                let value = lower_expression(context, function, value, None, values, emitter)?;
                values.insert(name.id, value);
            }
            Instruction::Return(Some(value)) => {
                return lower_expression(
                    context,
                    function,
                    value,
                    Some(return_type),
                    values,
                    emitter,
                )
                .map(Some);
            }
            Instruction::With { instructions, .. } => {
                if let Some(result) = lower_straight_line(
                    context,
                    function,
                    return_type,
                    instructions,
                    values,
                    emitter,
                )? {
                    return Ok(Some(result));
                }
            }
            Instruction::Assert(_) | Instruction::Evaluate(_) => {}
            other => {
                return Err(StableHloLoweringError::UnsupportedFunction {
                    function: function.into(),
                    reason: format!("unsupported tensor control-flow instruction {other:?}"),
                });
            }
        }
    }
    Ok(None)
}

fn lower_expression(
    context: &LoweringContext<'_>,
    function: &str,
    expression: &Expression,
    expected_type: Option<TensorType>,
    values: &HashMap<BindingId, MlirValue>,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    match expression.kind() {
        Expression::Variable(name) => values.get(&name.id).cloned().ok_or_else(|| {
            StableHloLoweringError::UnsupportedFunction {
                function: function.into(),
                reason: format!("unknown tensor SSA value `{name}`"),
            }
        }),
        Expression::Call { target, args } => {
            let result_type = match expected_type.or_else(|| match expression.ty() {
                Some(ValueType::Tensor(tensor)) if tensor.rank.is_some() => Some(tensor),
                _ => None,
            }) {
                Some(tensor) => tensor,
                other => {
                    return Err(StableHloLoweringError::UnsupportedFunction {
                        function: function.into(),
                        reason: format!(
                            "tensor call `{}` has non-tensor type {other:?}",
                            target.name
                        ),
                    });
                }
            };
            lower_call(
                context,
                function,
                target,
                args,
                result_type,
                values,
                emitter,
            )
        }
        other => Err(StableHloLoweringError::UnsupportedFunction {
            function: function.into(),
            reason: format!("unsupported tensor expression {other:?}"),
        }),
    }
}

fn lower_call(
    context: &LoweringContext<'_>,
    caller: &str,
    target: &CallTarget,
    arguments: &[Expression],
    result_type: TensorType,
    values: &HashMap<BindingId, MlirValue>,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    if let Some(function) = context.functions.get(&target.id).copied() {
        if function.native_symbol.is_none() {
            return lower_source_call(context, function, arguments, values, emitter);
        }
    }
    lower_intrinsic_call(
        context,
        caller,
        target,
        arguments,
        result_type,
        values,
        emitter,
    )
}

fn lower_source_call(
    context: &LoweringContext<'_>,
    function: &Function,
    arguments: &[Expression],
    caller_values: &HashMap<BindingId, MlirValue>,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    if function.params.len() != arguments.len() {
        return Err(StableHloLoweringError::InvalidArity {
            operation: function.name.clone(),
            expected: function.params.len(),
            actual: arguments.len(),
        });
    }
    let ValueType::Tensor(return_type) = function.return_type else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: function.name.clone(),
            reason: "an inlined tensor source call must return a tensor".into(),
        });
    };
    let mut callee_values = HashMap::new();
    for (parameter, argument) in function.params.iter().zip(arguments) {
        let ValueType::Tensor(parameter_type) = parameter.ty else {
            return Err(StableHloLoweringError::UnsupportedFunction {
                function: function.name.clone(),
                reason: format!("inlined parameter `{}` is not a tensor", parameter.name),
            });
        };
        let value = lower_expression(
            context,
            &function.name,
            argument,
            Some(parameter_type),
            caller_values,
            emitter,
        )?;
        callee_values.insert(parameter.name.id, value);
    }
    lower_straight_line(
        context,
        &function.name,
        return_type,
        &function.instructions,
        &mut callee_values,
        emitter,
    )?
    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
        function: function.name.clone(),
        reason: "inlined tensor function has no returned value".into(),
    })
}

fn lower_intrinsic_call(
    context: &LoweringContext<'_>,
    caller: &str,
    target: &CallTarget,
    arguments: &[Expression],
    result_type: TensorType,
    values: &HashMap<BindingId, MlirValue>,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let mut operands = Vec::new();
    for argument in arguments {
        if matches!(argument.ty(), Some(ValueType::Tensor(_))) {
            operands.push(lower_expression(
                context, caller, argument, None, values, emitter,
            )?);
        } else if let Expression::List(elements) = argument.kind() {
            for element in elements {
                if matches!(element.ty(), Some(ValueType::Tensor(_))) {
                    operands.push(lower_expression(
                        context, caller, element, None, values, emitter,
                    )?);
                }
            }
        }
    }
    lower_tensor_intrinsic(target, arguments, &operands, result_type, emitter)
}

fn lower_tensor_intrinsic(
    target: &CallTarget,
    source_args: &[Expression],
    args: &[MlirValue],
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let op = normalized_tensor_operation(&target.name);
    match op.as_str() {
        "gather" | "rankedgather" => {
            require_arity(&op, args, 2)?;
            let table = args[0]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let ids = args[1]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let table_rank =
                table
                    .rank
                    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                        function: target.name.clone(),
                        reason: "gather requires a ranked embedding table".into(),
                    })?;
            let index_rank =
                ids.rank
                    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                        function: target.name.clone(),
                        reason: "gather requires ranked indices".into(),
                    })?;
            if table_rank != 2 {
                return Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: 2,
                    actual: Some(table_rank),
                });
            }
            let TensorDimension::Static(vocabulary_size) = table.dimensions[0] else {
                return Err(StableHloLoweringError::UnsupportedOperation(
                    target.name.clone(),
                ));
            };
            let TensorDimension::Static(embedding_size) = table.dimensions[1] else {
                return Err(StableHloLoweringError::UnsupportedOperation(
                    target.name.clone(),
                ));
            };
            Ok(indexing::embedding_lookup(
                emitter,
                &args[0],
                &args[1],
                u64::from(index_rank),
                vocabulary_size,
                embedding_size,
                result_type,
            ))
        }
        "transpose" | "rankedpermute" => {
            require_arity(&op, args, 1)?;
            let axes = integer_list_argument(source_args.get(1), &target.name)?;
            Ok(emitter.transpose(&args[0], &axes, result_type))
        }
        "reshape" => {
            require_arity(&op, args, 1)?;
            let _shape = integer_list_argument(source_args.get(1), &target.name)?;
            Ok(emitter.reshape(&args[0], result_type))
        }
        "broadcast" => {
            require_arity(&op, args, 1)?;
            let _shape = integer_list_argument(source_args.get(1), &target.name)?;
            let input_rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            Ok(emitter.broadcast_in_dim(
                &args[0],
                &(0..u64::from(input_rank)).collect::<Vec<_>>(),
                result_type,
            ))
        }
        "scale" | "rankedscale" | "addscalar" => {
            require_arity(&op, args, 1)?;
            let literal = float_argument(source_args.get(1), &target.name)?;
            let scalar = emitter.splat(&literal, result_type);
            if op == "scale" || op == "rankedscale" {
                Ok(emitter.multiply(&args[0], &scalar, result_type))
            } else {
                Ok(emitter.add(&args[0], &scalar, result_type))
            }
        }
        "dynamicupdateslice" => {
            require_arity(&op, args, 2)?;
            let starts = integer_list_argument(source_args.get(2), &target.name)?
                .into_iter()
                .map(|start| emitter.scalar(&start.to_string(), TensorElementType::I64))
                .collect::<Vec<_>>();
            Ok(emitter.dynamic_update_slice(&args[0], &args[1], &starts, result_type))
        }
        "dynamicupdatesliceaxis" => {
            require_arity(&op, args, 3)?;
            let axis = integer_argument(source_args.get(3), &target.name)?;
            let rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            if axis >= u64::from(rank) {
                return Err(StableHloLoweringError::UnsupportedFunction {
                    function: target.name.clone(),
                    reason: format!("axis {axis} is outside rank {rank}"),
                });
            }
            let scalar_index = emitter.reshape(&args[2], scalar_tensor(TensorElementType::I64));
            let starts = (0..u64::from(rank))
                .map(|dimension| {
                    if dimension == axis {
                        scalar_index.clone()
                    } else {
                        emitter.scalar("0", TensorElementType::I64)
                    }
                })
                .collect::<Vec<_>>();
            Ok(emitter.dynamic_update_slice(&args[0], &args[1], &starts, result_type))
        }
        "dynamicslice" => {
            require_arity(&op, args, 1)?;
            let starts = integer_list_argument(source_args.get(1), &target.name)?
                .into_iter()
                .map(|start| emitter.scalar(&start.to_string(), TensorElementType::I64))
                .collect::<Vec<_>>();
            let sizes = integer_list_argument(source_args.get(2), &target.name)?;
            Ok(emitter.dynamic_slice(&args[0], &starts, &sizes, result_type))
        }
        "slice" => {
            require_arity(&op, args, 1)?;
            let starts = integer_list_argument(source_args.get(1), &target.name)?;
            let limits = integer_list_argument(source_args.get(2), &target.name)?;
            let strides = integer_list_argument(source_args.get(3), &target.name)?;
            Ok(emitter.slice(&args[0], &starts, &limits, &strides, result_type))
        }
        "cosine" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.cosine(&args[0], result_type))
        }
        "sine" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.sine(&args[0], result_type))
        }
        "concatenate" => {
            if args.is_empty() {
                return Err(StableHloLoweringError::InvalidArity {
                    operation: op,
                    expected: 1,
                    actual: 0,
                });
            }
            let axis = integer_argument(source_args.get(1), &target.name)?;
            Ok(emitter.concatenate(args, axis, result_type))
        }
        _ => lower_tensor_call(&target.name, args, result_type, emitter),
    }
}

fn normalized_tensor_operation(function: &str) -> String {
    let op = function
        .rsplit_once('.')
        .map(|(_, leaf)| leaf)
        .unwrap_or(function)
        .to_ascii_lowercase()
        .replace('_', "");
    op.strip_suffix("bf16")
        .or_else(|| op.strip_suffix("f32"))
        .unwrap_or(&op)
        .to_string()
}

fn integer_list_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<Vec<u64>, StableHloLoweringError> {
    let Some(Expression::List(values)) = expression.map(Expression::kind) else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a compile-time integer list".into(),
        });
    };
    values
        .iter()
        .map(|value| match value.kind() {
            Expression::Integer(value) if *value >= 0 => Ok(*value as u64),
            _ => Err(StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "expected non-negative compile-time integer metadata".into(),
            }),
        })
        .collect()
}

fn float_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<String, StableHloLoweringError> {
    match expression.map(Expression::kind) {
        Some(Expression::Float(bits)) => {
            let mut literal = f64::from_bits(*bits).to_string();
            if !literal.contains(['.', 'e', 'E']) {
                literal.push_str(".0");
            }
            Ok(literal)
        }
        Some(Expression::Integer(value)) => Ok(format!("{value}.0")),
        _ => Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a compile-time scalar".into(),
        }),
    }
}

fn integer_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<u64, StableHloLoweringError> {
    match expression.map(Expression::kind) {
        Some(Expression::Integer(value)) if *value >= 0 => Ok(*value as u64),
        _ => Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a non-negative compile-time integer".into(),
        }),
    }
}

pub fn lower_tensor_call(
    function: &str,
    args: &[MlirValue],
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let op = normalized_tensor_operation(function);

    match op.as_str() {
        "add" | "rankedadd" | "tensoradd" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.add(&args[0], &args[1], result_type))
        }

        "sub" | "subtract" | "rankedsubtract" | "tensorsub" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.subtract(&args[0], &args[1], result_type))
        }

        "mul" | "multiply" | "rankedmultiply" | "tensormul" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.multiply(&args[0], &args[1], result_type))
        }

        "div" | "divide" | "rankeddivide" | "tensordiv" => {
            require_arity(&op, args, 2)?;
            Ok(emitter.divide(&args[0], &args[1], result_type))
        }

        "matmul" | "batchedmatmul" | "rankedmatmul" | "tensormatmul" => {
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

        "reshape" | "tensorreshape" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.reshape(&args[0], result_type))
        }

        "rankedtranspose" => {
            require_arity(&op, args, 1)?;
            let rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let axes = (0..u64::from(rank)).rev().collect::<Vec<_>>();
            Ok(emitter.transpose(&args[0], &axes, result_type))
        }

        "broadcast" | "broadcastlike" | "broadcastindim" | "tensorbroadcast" => {
            if args.is_empty() || args.len() > 2 {
                return Err(StableHloLoweringError::InvalidArity {
                    operation: op,
                    expected: 1,
                    actual: args.len(),
                });
            }
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
            let input_type = args[0].tensor_type().unwrap();
            let width = usize::from(input_rank);
            let result_width = usize::from(result_rank);
            let first = (0..=result_width - width)
                .max_by_key(|&start| {
                    (0..width)
                        .filter(|&axis| {
                            input_type.dimensions[axis] == result_type.dimensions[start + axis]
                        })
                        .count()
                })
                .unwrap_or(result_width - width);
            let dimensions = (first..first + width)
                .map(|axis| axis as u64)
                .collect::<Vec<_>>();
            Ok(emitter.broadcast_in_dim(&args[0], &dimensions, result_type))
        }

        "relu" | "rankedrelu" | "tensorrelu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::relu(emitter, &args[0], result_type))
        }

        "silu" | "rankedsilu" | "swish" | "tensorsilu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::silu(emitter, &args[0], result_type))
        }

        "exp" | "rankedexp" | "exponential" | "tensorexp" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.exponential(&args[0], result_type))
        }

        "tanh" | "rankedtanh" | "tensortanh" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.tanh(&args[0], result_type))
        }

        "rsqrt" | "rankedrsqrt" | "tensorrsqrt" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.rsqrt(&args[0], result_type))
        }

        "sigmoid" | "rankedsigmoid" | "logistic" | "tensorsigmoid" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.logistic(&args[0], result_type))
        }

        "sum" | "rankedsum" | "sumlast" | "reducesum" | "tensorsum" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_sum(emitter, &args[0], &axes, result_type))
        }

        "max" | "maxlast" | "reducemax" | "tensormax" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_max(emitter, &args[0], &axes, result_type))
        }

        "min" | "reducemin" | "tensormin" => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_min(emitter, &args[0], &axes, result_type))
        }

        "mean" | "meanlast" | "reducemean" | "tensormean" => {
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

        "gelu" | "rankedgelu" | "tensorgelu" => {
            require_arity(&op, args, 1)?;
            Ok(activation::gelu_tanh(emitter, &args[0], result_type))
        }

        "softmax" | "rankedsoftmaxrows" | "softmaxlastaxis" | "tensorsoftmax" => {
            require_arity(&op, args, 1)?;
            let reduced_type = normalization::last_axis_reduced_type(result_type)?;
            Ok(normalization::softmax_last_axis(
                emitter,
                &args[0],
                result_type,
                reduced_type,
            ))
        }

        "rmsnorm" | "tensorrmsnorm" => {
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

        "layernorm" | "tensorlayernorm" => {
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

        "bf16to"
        | "f32to"
        | "tof8e4m3fn"
        | "tof8e5m2"
        | "tof16"
        | "tobf16"
        | "tof32"
        | "tof64"
        | "to"
        | "convert" => {
            require_arity(&op, args, 1)?;
            Ok(emitter.convert(&args[0], result_type))
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
    let result_rank =
        result_type
            .rank
            .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "reduction result is missing ranked tensor metadata".into(),
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
        let dimension = dimensions.get(axis as usize).ok_or_else(|| {
            StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: format!("axis {axis} is outside type {}", input.ty),
            }
        })?;
        let dimension =
            dimension
                .parse::<u64>()
                .map_err(|_| StableHloLoweringError::UnsupportedFunction {
                    function: operation.into(),
                    reason: "mean/norm requires static reduced dimensions".into(),
                })?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "reduction element count overflow".into(),
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
