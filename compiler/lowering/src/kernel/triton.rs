use super::{tensor_element_name, KernelBackend, KernelError, KernelIr};
use severian_mir::{ElementwiseKind, ReductionKind, TensorOp};

const ELEMENTWISE_BLOCK_SIZE: u32 = 256;
const REDUCTION_BLOCK_SIZE: u32 = 1024;

/// Host-independent launch metadata carried beside TTIR. Runtime integration
/// can consume this without importing Triton or Torch through Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TritonLaunch {
    pub entry: String,
    pub block_size: u32,
    pub programs: String,
    pub requires_zeroed_output: bool,
}

impl TritonLaunch {
    pub fn for_kernel(kernel: &KernelIr) -> Self {
        let (block_size, programs) = match &kernel.operation {
            TensorOp::Reduction(operation) if operation.kind == ReductionKind::Sum => {
                (REDUCTION_BLOCK_SIZE, "1".into())
            }
            TensorOp::Elementwise(operation) if operation.kind == ElementwiseKind::Relu => (
                ELEMENTWISE_BLOCK_SIZE,
                format!("ceil_div(element_count, {ELEMENTWISE_BLOCK_SIZE})"),
            ),
            _ => (ELEMENTWISE_BLOCK_SIZE, "1".into()),
        };
        Self {
            entry: kernel.name.clone(),
            block_size,
            programs,
            requires_zeroed_output: false,
        }
    }
}

/// Emits native Triton MLIR (TTIR). This is input to Triton's MLIR compiler,
/// not Python source and not a wrapper around Torch.
pub fn emit_triton_ir(kernel: &KernelIr) -> Result<String, KernelError> {
    kernel
        .triton_support()
        .map_err(|reason| KernelError::UnsupportedBackend {
            kernel: kernel.name.clone(),
            backend: KernelBackend::Triton,
            reason,
        })?;
    let input =
        kernel.parameters[kernel
            .input()
            .map_err(|reason| KernelError::UnsupportedBackend {
                kernel: kernel.name.clone(),
                backend: KernelBackend::Triton,
                reason,
            })?];
    let element = tensor_element_name(input.element);
    let launch = TritonLaunch::for_kernel(kernel);
    let body = match &kernel.operation {
        TensorOp::Reduction(operation) if operation.kind == ReductionKind::Sum => {
            reduction_sum_body(element, launch.block_size)
        }
        TensorOp::Elementwise(operation) if operation.kind == ElementwiseKind::Relu => {
            relu_body(element, launch.block_size)
        }
        _ => unreachable!("triton_support accepted an unsupported operation"),
    };
    Ok(format!(
        "module attributes {{\"severian.kernel\" = {entry:?}, \"severian.operation\" = {operation:?}, \"severian.launch.programs\" = {programs:?}, \"severian.launch.block_size\" = {block_size} : i32, \"severian.launch.requires_zeroed_output\" = {requires_zeroed_output}}} {{\n  tt.func public @{symbol}(%input: !tt.ptr<{element}>, %output: !tt.ptr<{element}>, %element_count: i32) attributes {{noinline = false}} {{\n{body}  }}\n}}\n",
        entry = kernel.name,
        operation = kernel.operation.name(),
        programs = launch.programs,
        block_size = launch.block_size,
        requires_zeroed_output = launch.requires_zeroed_output,
        symbol = sanitize_symbol(&kernel.name),
    ))
}

fn common_prefix(element: &str, block_size: u32) -> String {
    format!(
        "    %c{block}_i32 = arith.constant {block} : i32\n    %program = tt.get_program_id x : i32\n    %block_start = arith.muli %program, %c{block}_i32 : i32\n    %range = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>\n    %starts = tt.splat %block_start : i32 -> tensor<{block}xi32>\n    %offsets = arith.addi %starts, %range : tensor<{block}xi32>\n    %counts = tt.splat %element_count : i32 -> tensor<{block}xi32>\n    %mask = arith.cmpi slt, %offsets, %counts : tensor<{block}xi32>\n    %input_splat = tt.splat %input : !tt.ptr<{element}> -> tensor<{block}x!tt.ptr<{element}>>\n    %input_ptrs = tt.addptr %input_splat, %offsets : tensor<{block}x!tt.ptr<{element}>>, tensor<{block}xi32>\n",
        block = block_size,
    )
}

fn reduction_sum_body(element: &str, block_size: u32) -> String {
    format!(
        "    %c0_index = arith.constant 0 : index\n    %c1_index = arith.constant 1 : index\n    %c{block}_i32 = arith.constant {block} : i32\n    %c{last}_i32 = arith.constant {last} : i32\n    %zero_scalar = arith.constant 0.000000e+00 : {element}\n    %zero = arith.constant dense<0.000000e+00> : tensor<{block}x{element}>\n    %adjusted_count = arith.addi %element_count, %c{last}_i32 : i32\n    %block_count = arith.divsi %adjusted_count, %c{block}_i32 : i32\n    %block_count_index = arith.index_cast %block_count : i32 to index\n    %range = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>\n    %input_splat = tt.splat %input : !tt.ptr<{element}> -> tensor<{block}x!tt.ptr<{element}>>\n    %counts = tt.splat %element_count : i32 -> tensor<{block}xi32>\n    %total = scf.for %block_index = %c0_index to %block_count_index step %c1_index iter_args(%accumulator = %zero_scalar) -> ({element}) {{\n      %block_index_i32 = arith.index_cast %block_index : index to i32\n      %block_start = arith.muli %block_index_i32, %c{block}_i32 : i32\n      %starts = tt.splat %block_start : i32 -> tensor<{block}xi32>\n      %offsets = arith.addi %starts, %range : tensor<{block}xi32>\n      %mask = arith.cmpi slt, %offsets, %counts : tensor<{block}xi32>\n      %input_ptrs = tt.addptr %input_splat, %offsets : tensor<{block}x!tt.ptr<{element}>>, tensor<{block}xi32>\n      %values = tt.load %input_ptrs, %mask, %zero : tensor<{block}x!tt.ptr<{element}>>\n      %partial = \"tt.reduce\"(%values) <{{axis = 0 : i32}}> ({{\n      ^bb0(%left: {element}, %right: {element}):\n        %sum = arith.addf %left, %right : {element}\n        tt.reduce.return %sum : {element}\n      }}) : (tensor<{block}x{element}>) -> {element}\n      %next = arith.addf %accumulator, %partial : {element}\n      scf.yield %next : {element}\n    }}\n    tt.store %output, %total : !tt.ptr<{element}>\n    tt.return\n",
        block = block_size,
        last = block_size - 1,
    )
}

fn relu_body(element: &str, block_size: u32) -> String {
    let prefix = common_prefix(element, block_size);
    format!(
        "{prefix}    %zero = arith.constant dense<0.000000e+00> : tensor<{block}x{element}>\n    %values = tt.load %input_ptrs, %mask, %zero : tensor<{block}x!tt.ptr<{element}>>\n    %activated = arith.maximumf %values, %zero : tensor<{block}x{element}>\n    %output_splat = tt.splat %output : !tt.ptr<{element}> -> tensor<{block}x!tt.ptr<{element}>>\n    %output_ptrs = tt.addptr %output_splat, %offsets : tensor<{block}x!tt.ptr<{element}>>, tensor<{block}xi32>\n    tt.store %output_ptrs, %activated, %mask : tensor<{block}x!tt.ptr<{element}>>\n    tt.return\n",
        block = block_size,
    )
}

fn sanitize_symbol(name: &str) -> String {
    let symbol = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if symbol.is_empty() {
        "kernel".into()
    } else {
        symbol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_hir::{FunctionId, TensorDimension, TensorElementType, TensorType};
    use severian_mir::{ElementwiseOp, LocalId, ReductionOp, TensorOperand, ValueRef};

    fn operand(input: TensorType) -> TensorOperand {
        TensorOperand {
            value: ValueRef {
                id: None,
                ty: Some(severian_hir::ValueType::Tensor(input)),
                local: Some(LocalId(0)),
                tensor_op: None,
            },
            ty: input,
        }
    }

    fn kernel(operation: TensorOp) -> KernelIr {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        KernelIr {
            function: FunctionId::from_name("special"),
            name: "special".into(),
            parameters: vec![input],
            parameter_locals: vec![LocalId(0)],
            result: operation.result(),
            operation,
            policy: KernelBackend::Auto,
        }
    }

    #[test]
    fn reduction_is_native_ttir_with_launch_metadata() {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        let result = TensorType::ranked(TensorElementType::F32, &[]).unwrap();
        let source = emit_triton_ir(&kernel(TensorOp::Reduction(ReductionOp {
            kind: ReductionKind::Sum,
            input: operand(input),
            axes: vec![0],
            result,
        })))
        .unwrap();
        assert!(source.contains("tt.func public @special"));
        assert!(source.contains("\"tt.reduce\""));
        assert!(source.contains("scf.for"));
        assert!(source.contains("tt.store %output, %total"));
        assert!(source.contains("\"severian.launch.programs\" = \"1\""));
        assert!(source.contains("\"severian.launch.requires_zeroed_output\" = false"));
        assert!(!source.contains("import "));
        assert!(!source.contains("python"));
        assert!(!source.contains("torch"));
    }

    #[test]
    fn relu_is_a_masked_elementwise_ttir_kernel() {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        let source = emit_triton_ir(&kernel(TensorOp::Elementwise(ElementwiseOp {
            kind: ElementwiseKind::Relu,
            inputs: vec![operand(input)],
            result: input,
        })))
        .unwrap();
        assert!(source.contains("arith.maximumf"));
        assert!(source.contains("tt.store"));
        assert!(!source.contains("scf.for"));
    }
}
