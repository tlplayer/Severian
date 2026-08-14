use crate::*;
use severian_hir::ValueType;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationError {
    pub function: String,
    pub block: Option<BlockId>,
    pub invariant: &'static str,
    pub message: String,
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MIR invariant `{}` failed", self.invariant)?;
        if let Some(block) = self.block {
            write!(formatter, " in {} block {}", self.function, block.0)?;
        } else {
            write!(formatter, " in {}", self.function)?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl std::error::Error for VerificationError {}

/// Verify the structural invariants that every MIR consumer may rely on.
///
/// This deliberately lives at the MIR boundary instead of in a backend. A
/// transformation that creates an invalid CFG is therefore blamed before
/// lowering can obscure the source of the defect.
pub fn verify(program: &Program) -> Result<(), Vec<VerificationError>> {
    let mut errors = Vec::new();
    let mut function_ids = BTreeSet::new();
    for function in &program.functions {
        if !function_ids.insert(function.id) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "unique-function-id",
                message: format!("stable function identity {:?} is reused", function.id),
            });
        }
        verify_function(function, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn verify_function(function: &Function, errors: &mut Vec<VerificationError>) {
    if function.blocks.is_empty() {
        errors.push(VerificationError {
            function: function.name.clone(),
            block: None,
            invariant: "entry-block",
            message: "function has no entry block".into(),
        });
        return;
    }

    let block_count = function.blocks.len();
    let local_count = function.locals.len();
    let tensor_operation_count = function.tensor_operations.len();
    if function.source_tensor_intrinsics != tensor_operation_count {
        errors.push(VerificationError {
            function: function.name.clone(),
            block: None,
            invariant: "complete-tensor-lowering",
            message: format!(
                "{} recognized tensor intrinsic(s) produced {tensor_operation_count} tensor operation(s)",
                function.source_tensor_intrinsics
            ),
        });
    }
    let mut binding_ids = BTreeSet::new();
    for (index, local) in function.locals.iter().enumerate() {
        if local.id.0 as usize != index {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "dense-local-id",
                message: format!("local at index {index} has id {}", local.id.0),
            });
        }
        if !binding_ids.insert(local.binding.id) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: None,
                invariant: "unique-binding-id",
                message: format!(
                    "binding identity {:?} is assigned to multiple MIR locals",
                    local.binding.id
                ),
            });
        }
    }
    for parameter in &function.parameters {
        verify_local_target(function, None, *parameter, local_count, errors);
    }
    for (operation_index, operation) in function.tensor_operations.iter().enumerate() {
        for input in operation.inputs() {
            verify_value_ref(
                function,
                None,
                input.value,
                local_count,
                tensor_operation_count,
                errors,
            );
            if input.value.ty != Some(ValueType::Tensor(input.ty)) {
                errors.push(VerificationError {
                    function: function.name.clone(),
                    block: None,
                    invariant: "tensor-operand-type",
                    message: format!(
                        "operation `{}` carries {:?}, but its value is typed {:?}",
                        operation.name(),
                        input.ty,
                        input.value.ty
                    ),
                });
            }
            if input
                .value
                .tensor_op
                .is_some_and(|dependency| dependency.0 as usize >= operation_index)
            {
                errors.push(VerificationError {
                    function: function.name.clone(),
                    block: None,
                    invariant: "tensor-operation-order",
                    message: format!(
                        "operation `{}` at index {operation_index} depends on a non-prior tensor operation",
                        operation.name()
                    ),
                });
            }
        }
    }
    for (index, block) in function.blocks.iter().enumerate() {
        if block.id.0 as usize != index {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "dense-block-id",
                message: format!("block at index {index} has id {}", block.id.0),
            });
        }
        if matches!(block.terminator, Terminator::Unreachable) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "terminated-block",
                message: "lowering left the block without a terminator".into(),
            });
        }
        for operation in &block.operations {
            if let OperationKind::Bind(local) | OperationKind::TryBind(local) = operation.kind {
                verify_local_target(function, Some(block.id), local, local_count, errors);
            }
            for operand in &operation.operands {
                verify_value_ref(
                    function,
                    Some(block.id),
                    *operand,
                    local_count,
                    tensor_operation_count,
                    errors,
                );
            }
        }
        for target in successor_blocks(&block.terminator) {
            if target.0 as usize >= block_count {
                errors.push(VerificationError {
                    function: function.name.clone(),
                    block: Some(block.id),
                    invariant: "valid-successor",
                    message: format!(
                        "terminator targets block {} but the function has {block_count} block(s)",
                        target.0
                    ),
                });
            }
        }
        for value in terminator_values(&block.terminator) {
            verify_value_ref(
                function,
                Some(block.id),
                value,
                local_count,
                tensor_operation_count,
                errors,
            );
        }
        verify_terminator_types(function, block, errors);
    }
}

fn verify_value_ref(
    function: &Function,
    block: Option<BlockId>,
    value: ValueRef,
    local_count: usize,
    tensor_operation_count: usize,
    errors: &mut Vec<VerificationError>,
) {
    if let Some(local) = value.local {
        verify_local_target(function, block, local, local_count, errors);
    }
    let Some(operation) = value.tensor_op else {
        return;
    };
    let Some(operation) = function.tensor_operations.get(operation.0 as usize) else {
        errors.push(VerificationError {
            function: function.name.clone(),
            block,
            invariant: "valid-tensor-operation",
            message: format!(
                "references tensor operation {} but the function has {tensor_operation_count} operation(s)",
                operation.0
            ),
        });
        return;
    };
    if value.ty != Some(ValueType::Tensor(operation.result())) {
        errors.push(VerificationError {
            function: function.name.clone(),
            block,
            invariant: "tensor-result-type",
            message: format!(
                "operation `{}` returns {:?}, but its value is typed {:?}",
                operation.name(),
                operation.result(),
                value.ty
            ),
        });
    }
}

fn verify_local_target(
    function: &Function,
    block: Option<BlockId>,
    local: LocalId,
    local_count: usize,
    errors: &mut Vec<VerificationError>,
) {
    if local.0 as usize >= local_count {
        errors.push(VerificationError {
            function: function.name.clone(),
            block,
            invariant: "valid-local",
            message: format!(
                "references local {} but the function has {local_count} local(s)",
                local.0
            ),
        });
    }
}

fn successor_blocks(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Goto(target) => vec![*target],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Loop { body, exit, .. } | Terminator::For { body, exit, .. } => {
            vec![*body, *exit]
        }
        Terminator::Switch { arms, exit, .. } => {
            let mut targets = arms.clone();
            targets.push(*exit);
            targets
        }
        Terminator::Return(_)
        | Terminator::Break
        | Terminator::Continue
        | Terminator::Unreachable => Vec::new(),
    }
}

fn terminator_values(terminator: &Terminator) -> Vec<ValueRef> {
    match terminator {
        Terminator::Return(value) => value.iter().copied().collect(),
        Terminator::Branch { condition, .. } | Terminator::Loop { condition, .. } => {
            vec![*condition]
        }
        Terminator::For { iterable, .. } => vec![*iterable],
        Terminator::Switch { values, .. } => values.clone(),
        Terminator::Goto(_)
        | Terminator::Break
        | Terminator::Continue
        | Terminator::Unreachable => Vec::new(),
    }
}

fn verify_terminator_types(
    function: &Function,
    block: &BasicBlock,
    errors: &mut Vec<VerificationError>,
) {
    let condition = match &block.terminator {
        Terminator::Branch { condition, .. } | Terminator::Loop { condition, .. } => {
            Some(condition)
        }
        _ => None,
    };
    if let Some(condition) = condition {
        if !matches!(condition.ty, Some(ValueType::Bool | ValueType::Any)) {
            errors.push(VerificationError {
                function: function.name.clone(),
                block: Some(block.id),
                invariant: "boolean-condition",
                message: format!("control-flow condition has type {:?}", condition.ty),
            });
        }
    }
}
