use super::{KernelBackend, KernelError, KernelIr, KernelOperation};
use severian_hir::{Expression, Function, Instruction, Program, TensorType, ValueType};

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
    if function.native_symbol.is_some() || function.params.is_empty() {
        return None;
    }
    let parameters = function
        .params
        .iter()
        .map(|parameter| match parameter.ty {
            ValueType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let ValueType::Tensor(declared_result) = function.return_type else {
        return None;
    };
    let operation = lower_body(&function.instructions, function)?;
    let result = match operation {
        KernelOperation::ReductionSum { .. } => TensorType::ranked(declared_result.element, &[])
            .expect("a scalar tensor is always representable"),
        KernelOperation::ElementwiseRelu { .. } => declared_result,
    };
    Some(KernelIr {
        function: function.id,
        name: function.name.clone(),
        parameters,
        result,
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

fn lower_body(instructions: &[Instruction], function: &Function) -> Option<KernelOperation> {
    let [instruction] = instructions else {
        return None;
    };
    match instruction {
        Instruction::Return(Some(value)) => lower_return(value, function),
        Instruction::With { instructions, .. } => lower_body(instructions, function),
        _ => None,
    }
}

fn lower_return(expression: &Expression, function: &Function) -> Option<KernelOperation> {
    let Expression::Call { target, args } = expression.kind() else {
        return None;
    };
    let operation = normalized_operation(&target.name);
    let expected_arity = match operation.as_str() {
        "sum" | "rankedsum" | "sumlast" | "reducesum" | "tensorsum" => 1,
        "relu" | "rankedrelu" | "tensorrelu" => 1,
        _ => return None,
    };
    if args.len() != expected_arity {
        return None;
    }
    let Expression::Variable(input_name) = args[0].kind() else {
        return None;
    };
    let input = function
        .params
        .iter()
        .position(|parameter| parameter.name == *input_name)?;
    match operation.as_str() {
        "sum" | "rankedsum" | "sumlast" | "reducesum" | "tensorsum" => {
            Some(KernelOperation::ReductionSum { input })
        }
        "relu" | "rankedrelu" | "tensorrelu" => Some(KernelOperation::ElementwiseRelu { input }),
        _ => None,
    }
}

fn normalized_operation(function: &str) -> String {
    let operation = function
        .rsplit_once('.')
        .map(|(_, name)| name)
        .unwrap_or(function)
        .to_ascii_lowercase()
        .replace('_', "");
    operation
        .strip_suffix("bf16")
        .or_else(|| operation.strip_suffix("f32"))
        .unwrap_or(&operation)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{
        CallTarget, FunctionId, Parameter, Program, TensorDimension, TensorElementType,
    };

    fn direct_program(operation: &str) -> Program {
        let tensor =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        Program {
            functions: vec![Function {
                id: FunctionId::from_name(operation),
                name: operation.into(),
                native_symbol: None,
                decorators: Vec::new(),
                contract: None,
                params: vec![Parameter {
                    name: "value".into(),
                    ty: ValueType::Tensor(tensor),
                    default: None,
                    receiver: None,
                }],
                return_type: ValueType::Tensor(tensor),
                instructions: vec![Instruction::Return(Some(Expression::Call {
                    target: CallTarget::source(format!("tensor.{operation}")),
                    args: vec![Expression::Variable("value".into())],
                }))],
                tests: Vec::new(),
            }],
            ..Program::default()
        }
    }

    #[test]
    fn recognizes_reduction_and_relu_regions() {
        assert!(matches!(
            find(&direct_program("sum"), None).unwrap().operation,
            KernelOperation::ReductionSum { .. }
        ));
        assert!(matches!(
            find(&direct_program("relu"), None).unwrap().operation,
            KernelOperation::ElementwiseRelu { .. }
        ));
    }

    #[test]
    fn does_not_discard_surrounding_source_operations() {
        let mut program = direct_program("sum");
        program.functions[0].instructions.insert(
            0,
            Instruction::Evaluate(Expression::Variable("value".into())),
        );
        assert!(collect(&program).is_empty());
    }
}
