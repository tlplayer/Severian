use super::abi::{mlir_type, task_type_suffix, LoweredValue};
use severian_hir::ValueType;

#[derive(Debug, Clone)]
pub struct AwaitLowering {
    pub result: Option<LoweredValue>,
    pub mlir: String,
}

pub fn emit_await(
    task_handle: &str,
    return_type: ValueType,
    result_name: Option<&str>,
) -> AwaitLowering {
    if return_type == ValueType::Unit {
        return AwaitLowering {
            result: None,
            mlir: format!(
                "    llvm.call @__sev_task_await_unit({task_handle}) : (!llvm.ptr) -> ()\n"
            ),
        };
    }

    let result_name = result_name.unwrap_or("%await_result");
    let suffix = task_type_suffix(return_type);
    let ty = mlir_type(return_type);

    AwaitLowering {
        result: Some(LoweredValue::new(result_name, return_type)),
        mlir: format!(
            "    {result_name} = llvm.call @__sev_task_await_{suffix}({task_handle}) : (!llvm.ptr) -> {ty}\n"
        ),
    }
}

pub fn emit_await_many<'a>(
    tasks: impl IntoIterator<Item = (&'a str, ValueType, Option<&'a str>)>,
) -> (Vec<LoweredValue>, String) {
    let mut results = Vec::new();
    let mut mlir = String::new();

    for (task, return_type, result_name) in tasks {
        let lowered = emit_await(task, return_type, result_name);
        mlir.push_str(&lowered.mlir);
        if let Some(result) = lowered.result {
            results.push(result);
        }
    }

    (results, mlir)
}
