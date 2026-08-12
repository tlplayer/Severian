//! Backend-neutral tensor-kernel planning and specialized backend emission.
//!
//! This is the boundary between Severian semantics and accelerator backends.
//! Benchmark adapters belong outside this module; emitted kernels expose a
//! small `launch` API and contain no knowledge of a particular harness.

use severian_hir::{
    Expression, Function, FunctionId, Instruction, Program, TensorElementType, TensorType,
    ValueType,
};
use severian_mlir::Module;
use std::fmt;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    Auto,
    Xla,
    Triton,
    Llvm,
}

impl KernelBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Xla => "xla",
            Self::Triton => "triton",
            Self::Llvm => "llvm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTarget {
    Cpu,
    Gpu,
    Tpu,
}

impl KernelTarget {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Tpu => "tpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOperation {
    ReductionSum { input: usize },
}

impl KernelOperation {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ReductionSum { .. } => "reduction.sum",
        }
    }

    pub const fn supports_triton(&self) -> bool {
        matches!(self, Self::ReductionSum { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIr {
    pub function: FunctionId,
    pub name: String,
    pub parameters: Vec<TensorType>,
    pub result: TensorType,
    pub operation: KernelOperation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSelection {
    pub requested: KernelBackend,
    pub selected: KernelBackend,
    pub target: KernelTarget,
    pub reason: String,
    pub fallback: Option<KernelBackend>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    NoKernels,
    UnknownEntry(String),
    AmbiguousEntries(Vec<String>),
    UnsupportedBackend {
        kernel: String,
        backend: KernelBackend,
        reason: String,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKernels => formatter.write_str(
                "no specialized tensor kernels were found; the first direct GPU lowering supports tensor reduction sum",
            ),
            Self::UnknownEntry(entry) => write!(formatter, "no kernel entry named `{entry}` was found"),
            Self::AmbiguousEntries(entries) => write!(
                formatter,
                "multiple kernel entries were found ({}); select one with `--entry`",
                entries.join(", ")
            ),
            Self::UnsupportedBackend {
                kernel,
                backend,
                reason,
            } => write!(
                formatter,
                "kernel `{kernel}` cannot use backend `{}`: {reason}",
                backend.name()
            ),
        }
    }
}

impl std::error::Error for KernelError {}

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

pub fn select_backend(
    kernel: &KernelIr,
    requested: KernelBackend,
    target: KernelTarget,
) -> Result<KernelSelection, KernelError> {
    let (selected, reason, fallback) = match requested {
        KernelBackend::Auto => match target {
            KernelTarget::Gpu if kernel.operation.supports_triton() => (
                KernelBackend::Triton,
                format!(
                    "recognized `{}` on a GPU target; direct kernel lowering is available",
                    kernel.operation.name()
                ),
                Some(KernelBackend::Xla),
            ),
            KernelTarget::Gpu => (
                KernelBackend::Xla,
                "no direct GPU lowering is available; use the portable tensor backend".into(),
                None,
            ),
            KernelTarget::Tpu => (
                KernelBackend::Xla,
                "TPU execution uses StableHLO and PJRT".into(),
                None,
            ),
            KernelTarget::Cpu => (
                KernelBackend::Llvm,
                "CPU execution uses the native LLVM path".into(),
                None,
            ),
        },
        KernelBackend::Triton => {
            if target != KernelTarget::Gpu {
                return Err(KernelError::UnsupportedBackend {
                    kernel: kernel.name.clone(),
                    backend: requested,
                    reason: "Triton kernels require a GPU target".into(),
                });
            }
            if !kernel.operation.supports_triton() {
                return Err(KernelError::UnsupportedBackend {
                    kernel: kernel.name.clone(),
                    backend: requested,
                    reason: format!(
                        "operation `{}` has no direct Triton lowering",
                        kernel.operation.name()
                    ),
                });
            }
            (
                KernelBackend::Triton,
                "the Triton backend was explicitly requested".into(),
                Some(KernelBackend::Xla),
            )
        }
        KernelBackend::Xla => (
            KernelBackend::Xla,
            "the portable StableHLO/XLA backend was explicitly requested".into(),
            None,
        ),
        KernelBackend::Llvm => {
            if target != KernelTarget::Cpu {
                return Err(KernelError::UnsupportedBackend {
                    kernel: kernel.name.clone(),
                    backend: requested,
                    reason: "the LLVM kernel path currently targets the CPU".into(),
                });
            }
            (
                KernelBackend::Llvm,
                "the native LLVM backend was explicitly requested".into(),
                None,
            )
        }
    };
    Ok(KernelSelection {
        requested,
        selected,
        target,
        reason,
        fallback,
    })
}

/// Emits a standalone Python module whose Triton frontend lowers the kernel to
/// TritonIR. The module deliberately exports only `launch`; benchmark-specific
/// call signatures are supplied by adapters outside the compiler.
pub fn emit_triton_python(kernel: &KernelIr) -> Result<String, KernelError> {
    if !kernel.operation.supports_triton() {
        return Err(KernelError::UnsupportedBackend {
            kernel: kernel.name.clone(),
            backend: KernelBackend::Triton,
            reason: format!(
                "operation `{}` has no direct Triton lowering",
                kernel.operation.name()
            ),
        });
    }
    let input = match kernel.operation {
        KernelOperation::ReductionSum { input } => input,
    };
    let element = tensor_element_name(kernel.parameters[input].element);
    Ok(format!(
        r#"# Generated by Severian. This is a standalone Triton kernel artifact.
# Source entry: {entry}
# Kernel IR operation: {operation}
import torch
import triton
import triton.language as tl

SEVERIAN_KERNEL_NAME = {entry:?}
SEVERIAN_KERNEL_OPERATION = {operation:?}
SEVERIAN_ELEMENT_TYPE = {element:?}

@triton.jit
def _severian_kernel(input_pointer, output_pointer, element_count: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_pointer + offsets, mask=offsets < element_count, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(output_pointer, partial)

def launch(input_tensor, output_tensor=None, block_size=1024):
    if output_tensor is None:
        output_tensor = torch.zeros((), device=input_tensor.device, dtype=input_tensor.dtype)
    else:
        output_tensor.zero_()
    element_count = input_tensor.numel()
    grid = (triton.cdiv(element_count, block_size),)
    _severian_kernel[grid](
        input_tensor,
        output_tensor,
        element_count,
        BLOCK_SIZE=block_size,
    )
    return output_tensor.reshape(-1)[0]
"#,
        entry = kernel.name,
        operation = kernel.operation.name(),
        element = element,
    ))
}

/// Emits the portable StableHLO representation from the same kernel IR used
/// by specialized backends. This keeps backend parity independent of legacy
/// source-level intrinsic aliases.
pub fn emit_stablehlo(kernel: &KernelIr) -> Result<Module, KernelError> {
    use crate::stablehlo::{reduction, MlirValue, StableHloEmitter};
    use crate::tensor::tensor_type;

    let input_index = match kernel.operation {
        KernelOperation::ReductionSum { input } => input,
    };
    let input = kernel.parameters.get(input_index).copied().ok_or_else(|| {
        KernelError::UnsupportedBackend {
            kernel: kernel.name.clone(),
            backend: KernelBackend::Xla,
            reason: format!("kernel input {input_index} does not exist"),
        }
    })?;
    let rank = input.rank.ok_or_else(|| KernelError::UnsupportedBackend {
        kernel: kernel.name.clone(),
        backend: KernelBackend::Xla,
        reason: "StableHLO requires ranked tensor metadata".into(),
    })?;
    let argument = MlirValue::from_tensor("%arg0", input);
    let mut emitter = StableHloEmitter::new();
    let value = match kernel.operation {
        KernelOperation::ReductionSum { .. } => reduction::reduce_sum(
            &mut emitter,
            &argument,
            &(0..u64::from(rank)).collect::<Vec<_>>(),
            kernel.result,
        ),
    };
    let mut output = String::from("module {\n");
    writeln!(
        output,
        "  func.func @main(%arg0: {}) -> {} {{",
        tensor_type(input),
        tensor_type(kernel.result)
    )
    .expect("writing to a String cannot fail");
    output.push_str(emitter.as_str());
    writeln!(output, "    return {} : {}", value.name, value.ty)
        .expect("writing to a String cannot fail");
    output.push_str("  }\n}\n");
    Ok(Module::new(output))
}

fn lower_function(function: &Function) -> Option<KernelIr> {
    if function.native_symbol.is_some() || function.params.len() != 1 {
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
    };
    Some(KernelIr {
        function: function.id,
        name: function.name.clone(),
        parameters,
        result,
        operation,
    })
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
    let operation = target
        .name
        .rsplit_once('.')
        .map(|(_, name)| name)
        .unwrap_or(&target.name)
        .to_ascii_lowercase();
    let operation = operation
        .strip_suffix("bf16")
        .or_else(|| operation.strip_suffix("f32"))
        .unwrap_or(&operation);
    if !matches!(
        operation,
        "sum" | "rankedsum" | "sumlast" | "reduce_sum" | "tensor_sum"
    ) || args.len() != 1
    {
        return None;
    }
    let Expression::Variable(input_name) = args[0].kind() else {
        return None;
    };
    let input = function
        .params
        .iter()
        .position(|parameter| parameter.name == *input_name)?;
    Some(KernelOperation::ReductionSum { input })
}

const fn tensor_element_name(element: TensorElementType) -> &'static str {
    match element {
        TensorElementType::BF16 => "bf16",
        TensorElementType::F32 => "f32",
        TensorElementType::F64 => "f64",
        TensorElementType::I32 => "i32",
        TensorElementType::I64 => "i64",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{CallTarget, Parameter, TensorDimension};

    fn tensor() -> TensorType {
        TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap()
    }

    fn reduction_program() -> Program {
        let tensor = tensor();
        Program {
            functions: vec![Function {
                id: FunctionId::from_name("reduce"),
                name: "reduce".into(),
                native_symbol: None,
                decorators: Vec::new(),
                contract: None,
                params: vec![Parameter {
                    name: "value".into(),
                    ty: ValueType::Tensor(tensor),
                    default: None,
                }],
                return_type: ValueType::Tensor(tensor),
                instructions: vec![Instruction::Return(Some(Expression::Call {
                    target: CallTarget::source("tensor.sum"),
                    args: vec![Expression::Variable("value".into())],
                }))],
                tests: Vec::new(),
            }],
            ..Program::default()
        }
    }

    #[test]
    fn automatic_gpu_selection_uses_triton_with_xla_fallback() {
        let kernel = find(&reduction_program(), None).unwrap();
        let selection = select_backend(&kernel, KernelBackend::Auto, KernelTarget::Gpu).unwrap();
        assert_eq!(selection.selected, KernelBackend::Triton);
        assert_eq!(selection.fallback, Some(KernelBackend::Xla));
    }

    #[test]
    fn emitted_module_has_no_benchmark_protocol() {
        let kernel = find(&reduction_program(), None).unwrap();
        let source = emit_triton_python(&kernel).unwrap();
        assert!(source.contains("def launch("));
        assert!(source.contains("@triton.jit"));
        assert!(!source.contains("custom_kernel"));
        assert!(!source.contains("from task import"));
    }

    #[test]
    fn portable_emission_uses_the_same_kernel_ir() {
        let kernel = find(&reduction_program(), None).unwrap();
        let source = emit_stablehlo(&kernel).unwrap();
        assert!(source.as_str().contains("stablehlo.reduce"));
        assert!(source.as_str().contains("-> tensor<f32>"));
    }

    #[test]
    fn does_not_discard_surrounding_source_operations() {
        let mut program = reduction_program();
        program.functions[0].instructions.insert(
            0,
            Instruction::Evaluate(Expression::Variable("value".into())),
        );
        assert!(collect(&program).is_empty());
    }
}
