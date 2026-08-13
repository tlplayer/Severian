use super::{tensor_element_name, KernelBackend, KernelError, KernelIr, KernelOperation};

const BLOCK_SIZE: u32 = 256;

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
        Self {
            entry: kernel.name.clone(),
            block_size: BLOCK_SIZE,
            programs: format!("ceil_div(element_count, {BLOCK_SIZE})"),
            requires_zeroed_output: matches!(
                kernel.operation,
                KernelOperation::ReductionSum { .. }
            ),
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
    let input = kernel.parameters[kernel.operation.input()];
    let element = tensor_element_name(input.element);
    let launch = TritonLaunch::for_kernel(kernel);
    let body = match kernel.operation {
        KernelOperation::ReductionSum { .. } => reduction_sum_body(element),
        KernelOperation::ElementwiseRelu { .. } => relu_body(element),
    };
    Ok(format!(
        "module attributes {{severian_kernel = {entry:?}, severian_operation = {operation:?}, severian_launch_programs = {programs:?}, severian_launch_block_size = {block_size} : i32, severian_launch_requires_zeroed_output = {requires_zeroed_output}}} {{\n  tt.func public @{symbol}(%input: !tt.ptr<{element}>, %output: !tt.ptr<{element}>, %element_count: i32) attributes {{noinline = false}} {{\n{body}  }}\n}}\n",
        entry = kernel.name,
        operation = kernel.operation.name(),
        programs = launch.programs,
        block_size = launch.block_size,
        requires_zeroed_output = launch.requires_zeroed_output,
        symbol = sanitize_symbol(&kernel.name),
    ))
}

fn common_prefix(element: &str) -> String {
    format!(
        "    %c{block}_i32 = arith.constant {block} : i32\n    %program = tt.get_program_id x : i32\n    %block_start = arith.muli %program, %c{block}_i32 : i32\n    %range = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>\n    %starts = tt.splat %block_start : i32 -> tensor<{block}xi32>\n    %offsets = arith.addi %starts, %range : tensor<{block}xi32>\n    %counts = tt.splat %element_count : i32 -> tensor<{block}xi32>\n    %mask = arith.cmpi slt, %offsets, %counts : tensor<{block}xi32>\n    %input_splat = tt.splat %input : !tt.ptr<{element}> -> tensor<{block}x!tt.ptr<{element}>>\n    %input_ptrs = tt.addptr %input_splat, %offsets : tensor<{block}x!tt.ptr<{element}>>, tensor<{block}xi32>\n",
        block = BLOCK_SIZE,
    )
}

fn reduction_sum_body(element: &str) -> String {
    let prefix = common_prefix(element);
    format!(
        "{prefix}    %zero = arith.constant dense<0.000000e+00> : tensor<{block}x{element}>\n    %values = tt.load %input_ptrs, %mask, %zero : tensor<{block}x!tt.ptr<{element}>>\n    %partial = \"tt.reduce\"(%values) <{{axis = 0 : i32}}> ({{\n    ^bb0(%left: {element}, %right: {element}):\n      %sum = arith.addf %left, %right : {element}\n      tt.reduce.return %sum : {element}\n    }}) : (tensor<{block}x{element}>) -> {element}\n    %old = tt.atomic_rmw fadd, acq_rel, gpu, %output, %partial : (!tt.ptr<{element}>, {element}) -> {element}\n    tt.return\n",
        block = BLOCK_SIZE,
    )
}

fn relu_body(element: &str) -> String {
    let prefix = common_prefix(element);
    format!(
        "{prefix}    %zero = arith.constant dense<0.000000e+00> : tensor<{block}x{element}>\n    %values = tt.load %input_ptrs, %mask, %zero : tensor<{block}x!tt.ptr<{element}>>\n    %activated = arith.maximumf %values, %zero : tensor<{block}x{element}>\n    %output_splat = tt.splat %output : !tt.ptr<{element}> -> tensor<{block}x!tt.ptr<{element}>>\n    %output_ptrs = tt.addptr %output_splat, %offsets : tensor<{block}x!tt.ptr<{element}>>, tensor<{block}xi32>\n    tt.store %output_ptrs, %activated, %mask : tensor<{block}x!tt.ptr<{element}>>\n    tt.return\n",
        block = BLOCK_SIZE,
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

    fn kernel(operation: KernelOperation) -> KernelIr {
        let input =
            TensorType::ranked(TensorElementType::F32, &[TensorDimension::Dynamic]).unwrap();
        KernelIr {
            function: FunctionId::from_name("special"),
            name: "special".into(),
            parameters: vec![input],
            result: input,
            operation,
            policy: KernelBackend::Auto,
        }
    }

    #[test]
    fn reduction_is_native_ttir_with_launch_metadata() {
        let source = emit_triton_ir(&kernel(KernelOperation::ReductionSum { input: 0 })).unwrap();
        assert!(source.contains("tt.func public @special"));
        assert!(source.contains("\"tt.reduce\""));
        assert!(source.contains("tt.atomic_rmw fadd"));
        assert!(source.contains("severian_launch_programs"));
        assert!(source.contains("severian_launch_requires_zeroed_output = true"));
        assert!(!source.contains("import "));
        assert!(!source.contains("python"));
        assert!(!source.contains("torch"));
    }

    #[test]
    fn relu_is_a_masked_elementwise_ttir_kernel() {
        let source =
            emit_triton_ir(&kernel(KernelOperation::ElementwiseRelu { input: 0 })).unwrap();
        assert!(source.contains("arith.maximumf"));
        assert!(source.contains("tt.store"));
        assert!(!source.contains("tt.atomic_rmw"));
    }
}
