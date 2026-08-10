use crate::lowering_abi::{mlir_type, LoweredValue};
use severian_hir::{TaskPlacement, ValueType};

#[derive(Debug, Clone)]
pub struct TaskSpawnSpec {
    pub function: String,
    pub arguments: Vec<LoweredValue>,
    pub return_type: ValueType,
    pub placement: TaskPlacement,
}

#[derive(Debug, Clone)]
pub struct TaskSpawnLowering {
    pub result: LoweredValue,
    pub return_type: ValueType,
    pub mlir: String,
}

pub fn emit_task_spawn(
    result_name: impl Into<String>,
    spec: &TaskSpawnSpec,
) -> TaskSpawnLowering {
    let result_name = result_name.into();

    let values = spec
        .arguments
        .iter()
        .map(|argument| argument.value.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let types = spec
        .arguments
        .iter()
        .map(|argument| mlir_type(argument.ty))
        .collect::<Vec<_>>()
        .join(", ");

    let attributes = placement_attributes(spec.placement);

    let mlir = format!(
        "    {result_name} = llvm.call @__sev_task_spawn_{}({values}){attributes} : ({types}) -> !llvm.ptr\n",
        spec.function
    );

    TaskSpawnLowering {
        result: LoweredValue::new(result_name, ValueType::Any),
        return_type: spec.return_type,
        mlir,
    }
}

pub fn placement_attributes(placement: TaskPlacement) -> String {
    let attributes: &[&str] = match placement {
        TaskPlacement::Default => &[],
        TaskPlacement::Local => &["severian_distribution = \"local\""],
        TaskPlacement::Gpu => &[
            "severian_parallel = \"gpu\"",
            "severian_device_fallback = \"cpu\"",
        ],
        TaskPlacement::Simd => &[
            "severian_parallel = \"simd\"",
            "severian_device_fallback = \"cpu\"",
        ],
        TaskPlacement::Simt => &[
            "severian_parallel = \"simt\"",
            "severian_device_fallback = \"cpu\"",
        ],
    };

    if attributes.is_empty() {
        String::new()
    } else {
        format!(" {{{}}}", attributes.join(", "))
    }
}

pub fn task_spawn_declaration(
    function: &str,
    parameter_types: &[ValueType],
) -> String {
    let parameters = parameter_types
        .iter()
        .copied()
        .map(mlir_type)
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "  llvm.func @__sev_task_spawn_{function}({parameters}) -> !llvm.ptr\n"
    )
}
