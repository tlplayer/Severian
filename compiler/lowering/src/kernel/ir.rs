use super::{KernelBackend, KernelError, KernelIr};
use severian_hir::ValueType;
use severian_mir::{Function, OperationKind, Program};

pub fn collect(program: &Program) -> Vec<KernelIr> {
    program
        .functions
        .iter()
        .filter_map(lower_function)
        .collect()
}

pub fn find(program: &Program, entry: Option<&str>) -> Result<KernelIr, KernelError> {
    let kernels = collect(program);
    if let Some(entry) = entry {
        return kernels
            .into_iter()
            .find(|kernel| kernel.name == entry)
            .ok_or_else(|| KernelError::UnknownEntry(entry.into()));
    }
    match kernels.as_slice() {
        [] => Err(KernelError::NoKernels),
        [kernel] => Ok(kernel.clone()),
        _ => Err(KernelError::AmbiguousEntries(
            kernels.into_iter().map(|kernel| kernel.name).collect(),
        )),
    }
}

fn lower_function(function: &Function) -> Option<KernelIr> {
    if function.native_symbol.is_some() || function.parameters.is_empty() {
        return None;
    }
    let parameters = function
        .parameters
        .iter()
        .map(
            |parameter| match function.locals.get(parameter.0 as usize)?.ty {
                ValueType::Tensor(tensor) => Some(tensor),
                _ => None,
            },
        )
        .collect::<Option<Vec<_>>>()?;
    let [block] = function.blocks.as_slice() else {
        return None;
    };
    if !block
        .operations
        .iter()
        .all(|operation| matches!(operation.kind, OperationKind::With))
    {
        return None;
    }
    let severian_mir::Terminator::Return(Some(value)) = block.terminator else {
        return None;
    };
    let operation = function
        .tensor_operations
        .get(value.tensor_op?.0 as usize)?
        .clone();
    for input in operation.inputs() {
        if input.value.tensor_op.is_some() {
            return None;
        }
        let local = input.value.local?;
        if !function.parameters.contains(&local) {
            return None;
        }
    }
    Some(KernelIr {
        function: function.id,
        name: function.name.clone(),
        parameter_locals: function.parameters.clone(),
        parameters,
        result: operation.result(),
        operation,
        policy: compile_policy(function),
    })
}

fn compile_policy(function: &Function) -> KernelBackend {
    let Some(policy) = function
        .decorators
        .iter()
        .find(|decorator| decorator.package == "compile")
        .and_then(|decorator| decorator.symbols.first())
    else {
        return KernelBackend::Auto;
    };
    match policy.as_str() {
        "xla" => KernelBackend::Xla,
        "triton" => KernelBackend::Triton,
        _ => KernelBackend::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{
        BindingRef, CallTarget, Expression, Function as HirFunction, FunctionId, Instruction,
        Parameter, Program as HirProgram, TensorDimension, TensorElementType, TensorType,
    };
    use severian_mir::{ElementwiseKind, ReductionKind, TensorOp};

    fn typed(ty: ValueType, expression: Expression) -> Expression {
        Expression::Typed {
            id: severian_hir::HirId::synthetic(1),
            ty,
            any_origin: None,
            expression: Box::new(expression),
        }
    }

    fn direct_program(symbol: &str, result: TensorType) -> Program {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        let binding = BindingRef::synthetic("value");
        let call = typed(
            ValueType::Tensor(result),
            Expression::Call {
                target: CallTarget::native(symbol, symbol),
                args: vec![typed(
                    ValueType::Tensor(input),
                    Expression::Variable(binding.clone()),
                )],
            },
        );
        severian_mir::lower(&HirProgram {
            functions: vec![HirFunction {
                id: FunctionId::from_name(symbol),
                name: symbol.into(),
                native_symbol: None,
                decorators: Vec::new(),
                contract: None,
                params: vec![Parameter {
                    name: binding,
                    ty: ValueType::Tensor(input),
                    default: None,
                    receiver: None,
                }],
                return_type: ValueType::Tensor(result),
                instructions: vec![Instruction::Return(Some(call))],
                tests: Vec::new(),
            }],
            ..HirProgram::default()
        })
    }

    #[test]
    fn consumes_resolved_mir_reduction_and_relu_operations() {
        let scalar = TensorType::ranked(TensorElementType::F32, &[]).unwrap();
        assert!(matches!(
            find(&direct_program("__sev_tensor_sum", scalar), None)
                .unwrap()
                .operation,
            TensorOp::Reduction(operation) if operation.kind == ReductionKind::Sum
        ));

        let vector =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        assert!(matches!(
            find(&direct_program("__sev_tensor_relu", vector), None)
                .unwrap()
                .operation,
            TensorOp::Elementwise(operation) if operation.kind == ElementwiseKind::Relu
        ));
    }

    #[test]
    fn does_not_discard_surrounding_source_operations() {
        let scalar = TensorType::ranked(TensorElementType::F32, &[]).unwrap();
        let mut program = direct_program("__sev_tensor_sum", scalar);
        program.functions[0].blocks[0].operations.insert(
            0,
            severian_mir::Operation {
                kind: OperationKind::Evaluate,
                operands: Vec::new(),
            },
        );
        assert!(collect(&program).is_empty());
    }
}
