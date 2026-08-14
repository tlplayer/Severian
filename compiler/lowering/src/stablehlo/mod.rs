//! StableHLO emission for XLA-bound tensor regions.
//!
//! Severian's lowering crate currently emits textual MLIR. This module follows
//! the same model and gives the compiler a small typed layer over StableHLO
//! instead of scattering operation syntax through the main lowering file.

pub mod activation;
pub mod attention;
pub mod convolution;
pub mod indexing;
mod intrinsic;
pub mod linear;
pub mod normalization;
pub mod ops;
pub mod reduction;

pub use intrinsic::argument;
pub(crate) use intrinsic::scalar_tensor;
pub use ops::{MlirValue, StableHloEmitter};
pub use reduction::StableHloReduction;

use severian_hir::TensorType;
use severian_hir::{
    BindingId, CallTarget, Expression, Function, FunctionId, Instruction, Program, ValueType,
};
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
    intrinsic::lower(target, arguments, &operands, result_type, emitter)
}

#[cfg(test)]
mod tests;
