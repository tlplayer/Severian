use super::{KernelBackend, KernelError, KernelIr};
use severian_mir::{ElementwiseKind, ReductionKind, TensorOp};
use severian_mlir::Module;
use std::fmt::Write;

/// Emits the portable StableHLO representation from the same kernel IR used
/// by specialized backends.
pub fn emit_stablehlo(kernel: &KernelIr) -> Result<Module, KernelError> {
    use crate::stablehlo::{activation, reduction, MlirValue, StableHloEmitter};
    use crate::tensor::tensor_type;

    let input_index = kernel
        .input()
        .map_err(|reason| KernelError::UnsupportedBackend {
            kernel: kernel.name.clone(),
            backend: KernelBackend::Xla,
            reason,
        })?;
    let input = kernel.parameters.get(input_index).copied().ok_or_else(|| {
        KernelError::UnsupportedBackend {
            kernel: kernel.name.clone(),
            backend: KernelBackend::Xla,
            reason: format!("kernel input {input_index} does not exist"),
        }
    })?;
    let argument = MlirValue::from_tensor(format!("%arg{input_index}"), input);
    let mut emitter = StableHloEmitter::new();
    let value = match &kernel.operation {
        TensorOp::Reduction(operation) if operation.kind == ReductionKind::Sum => {
            reduction::reduce_sum(&mut emitter, &argument, &operation.axes, kernel.result)
        }
        TensorOp::Elementwise(operation) if operation.kind == ElementwiseKind::Relu => {
            activation::relu(&mut emitter, &argument, kernel.result)
        }
        operation => {
            return Err(KernelError::UnsupportedBackend {
                kernel: kernel.name.clone(),
                backend: KernelBackend::Xla,
                reason: format!(
                    "operation `{}` has no outlined StableHLO lowering",
                    operation.name()
                ),
            })
        }
    };
    let parameters = kernel
        .parameters
        .iter()
        .enumerate()
        .map(|(index, tensor)| format!("%arg{index}: {}", tensor_type(*tensor)))
        .collect::<Vec<_>>()
        .join(", ");
    let mut output = String::from("module {\n");
    writeln!(
        output,
        "  func.func @main({parameters}) -> {} {{",
        tensor_type(kernel.result)
    )
    .expect("writing to a String cannot fail");
    output.push_str(emitter.as_str());
    writeln!(output, "    return {} : {}", value.name, value.ty)
        .expect("writing to a String cannot fail");
    output.push_str("  }\n}\n");
    Ok(Module::new(output))
}
