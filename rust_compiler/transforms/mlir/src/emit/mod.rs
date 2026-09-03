use severian_artifact::ArtifactId;
use severian_lir::{
    BinaryOperation, Block, Constant, Function, FunctionId, FunctionLinkage, LoweredFloatFormat,
    LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape, LoweredType, Module,
    Operation, UnaryOperation, ValueId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirArtifact {
    /// A complete MLIR module containing exactly one entry function.
    pub module: String,
    pub inputs: Vec<LoweredType>,
    pub outputs: Vec<LoweredType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirError {
    DialectNotAllowed {
        dialect: String,
        target: String,
    },
    DuplicateSymbol(String),
    ArtifactSignatureConflict(ArtifactId),
    EntryFunctionCount(usize),
    EntryFunctionIsDeclaration,
    InvalidValue(ValueId),
    MlirApi(String),
    ParseFailed(String),
    SignatureMismatch,
    TargetMismatch {
        artifact: String,
        composition: String,
    },
    UnsupportedType(LoweredType),
    UnsupportedOperation(String),
    VerificationFailed(String),
}

impl fmt::Display for MlirError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MlirError {}

pub fn render(module: &Module) -> Result<String, MlirError> {
    if module.initializer_cfg.is_some() {
        return render_cfg_module(module);
    }
    let mut artifact_signatures =
        BTreeMap::<ArtifactId, (Vec<LoweredType>, Vec<LoweredType>)>::new();
    let mut runtime_signatures = BTreeMap::<String, (Vec<LoweredType>, Option<LoweredType>)>::new();
    let uses_task_lock = all_operations(module).into_iter().any(|operation| {
        matches!(
            operation,
            Operation::Spawn { locked: true, .. }
                | Operation::SpawnFieldUpdate { locked: true, .. }
        )
    });
    for operation in all_operations(module) {
        match operation {
            Operation::ArtifactCall {
                artifact,
                inputs,
                outputs,
            } => {
                let inputs = inputs
                    .iter()
                    .map(|value| value_type(module, *value))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = outputs
                    .iter()
                    .map(|value| value_type(module, *value))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some((known_inputs, known_outputs)) = artifact_signatures.get(artifact) {
                    if known_inputs != &inputs || known_outputs != &outputs {
                        return Err(MlirError::ArtifactSignatureConflict(*artifact));
                    }
                } else {
                    artifact_signatures.insert(*artifact, (inputs, outputs));
                }
            }
            Operation::RuntimeCall {
                symbol,
                arguments,
                result,
            } => {
                let inputs = arguments
                    .iter()
                    .map(|value| value_type(module, *value))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = result.map(|value| value_type(module, value)).transpose()?;
                if inputs
                    .iter()
                    .chain(result.iter())
                    .any(|ty| matches!(ty, LoweredType::Tensor { .. }))
                {
                    return Err(MlirError::UnsupportedOperation(format!(
                        "native runtime symbol `{symbol}` cannot receive or return an MLIR builtin tensor; use a !llvm.ptr StorageViewAbi at the host boundary and specialize it into a ranked tensor before compute lowering"
                    )));
                }
                if let Some(known) =
                    runtime_signatures.insert(symbol.clone(), (inputs.clone(), result.clone()))
                {
                    if known != (inputs, result) {
                        return Err(MlirError::UnsupportedOperation(format!(
                            "runtime symbol `{symbol}` has conflicting physical signatures"
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    let coverage = all_operations(module)
        .into_iter()
        .filter_map(|operation| match operation {
            Operation::Coverage { key } => Some(key),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    let mut output = String::new();
    render_class_aliases(&mut output, module)?;
    if let Some(architecture) = &module.gpu_architecture {
        output.push_str(&format!(
            "module attributes {{severian.gpu.architecture = \"{}\"}} {{\n",
            mlir_string(architecture)
        ));
    } else {
        output.push_str("module {\n");
    }
    output.push_str("  func.func private @__sev_process_set_arguments(i32, !llvm.ptr)\n");
    output
        .push_str("  func.func private @__sev_tensor_box_unranked(i64, !llvm.ptr) -> !llvm.ptr\n");
    output.push_str("  func.func private @__sev_tensor_box_rank(!llvm.ptr) -> i64\n");
    output.push_str("  func.func private @__sev_tensor_box_descriptor(!llvm.ptr) -> !llvm.ptr\n");
    if !runtime_signatures.contains_key("__sev_tensor_local_release") {
        output.push_str("  func.func private @__sev_tensor_local_release(!llvm.ptr)\n");
    }
    if uses_task_lock {
        output.push_str("  func.func private @__sev_task_lock()\n");
        output.push_str("  func.func private @__sev_task_unlock()\n");
    }
    for declaration in &module.traits {
        output.push_str(&format!(
            "  // severian trait @{} [{:032x}:{:032x}:{:032x}] (compile-time only)\n",
            declaration.name,
            declaration.id.package,
            declaration.id.module,
            declaration.id.declaration
        ));
        for method in &declaration.methods {
            let parameters = method
                .parameters
                .iter()
                .map(mlir_trait_type)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            output.push_str(&format!(
                "  //   def @{}({}) -> {}\n",
                method.name,
                parameters,
                mlir_trait_type(&method.result)?
            ));
        }
    }
    let mut declared_external_symbols = BTreeSet::new();
    let uses_aggregate_runtime = runtime_signatures.iter().any(|(symbol, (inputs, result))| {
        symbol.contains("_aggregate")
            && (inputs
                .iter()
                .any(|ty| matches!(ty, LoweredType::Aggregate(_)))
                || result
                    .as_ref()
                    .is_some_and(|ty| matches!(ty, LoweredType::Aggregate(_))))
    });
    if uses_aggregate_runtime {
        output.push_str("  func.func private @__sev_aggregate_box(!llvm.ptr, i64) -> !llvm.ptr\n");
    }
    for (symbol, (inputs, result)) in runtime_signatures {
        let aggregate_abi = symbol.contains("_aggregate");
        let inputs = inputs
            .into_iter()
            .map(|ty| runtime_abi_type(ty, aggregate_abi))
            .map(|ty| mlir_type(&ty))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let result = result
            .map(|ty| runtime_abi_type(ty, aggregate_abi))
            .map(|ty| mlir_type(&ty))
            .transpose()?
            .map(|result| format!(" -> {result}"))
            .unwrap_or_default();
        output.push_str(&format!(
            "  func.func private @{symbol}({inputs}){result}\n"
        ));
        declared_external_symbols.insert(symbol);
    }
    if !coverage.is_empty() {
        output.push_str("  func.func private @__sev_coverage_hit(!llvm.ptr)\n");
        for key in &coverage {
            output.push_str(&format!(
                "  llvm.mlir.global private constant @{}(\"{}\\00\") : !llvm.array<{} x i8>\n",
                coverage_symbol(key),
                mlir_string(key),
                key.len() + 1,
            ));
        }
    }
    for operation in all_operations(module) {
        if let Operation::Constant {
            value: Constant::String(value),
            result,
        } = operation
        {
            output.push_str(&format!(
                "  llvm.mlir.global private constant @{}(\"{}\\00\") : !llvm.array<{} x i8>\n",
                string_symbol(*result),
                mlir_string(value),
                value.len() + 1,
            ));
        }
    }
    for (artifact, (inputs, outputs)) in artifact_signatures {
        let symbol = artifact_symbol(artifact);
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(index, ty)| artifact_parameter(index, &ty))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let outputs = outputs
            .into_iter()
            .map(|ty| mlir_type(&ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result = match outputs.as_slice() {
            [] => String::new(),
            [output_type] => format!(" -> {output_type}"),
            output_types => format!(" -> ({})", output_types.join(", ")),
        };
        output.push_str(&format!(
            "  func.func private @{symbol}({inputs}){result}\n"
        ));
    }
    for function in module
        .functions
        .iter()
        .filter(|function| matches!(function.linkage, FunctionLinkage::External { .. }))
    {
        if declared_external_symbols.insert(function_symbol(function)) {
            render_function_declaration(&mut output, module, function)?;
        }
    }
    for function in module
        .functions
        .iter()
        .filter(|function| function.body.is_some())
    {
        render_function_definition(&mut output, module, function)?;
    }
    output.push_str("  func.func @main() -> i32 {\n");
    let mut coverage_ordinal = 0;
    render_block(
        &mut output,
        module,
        &module.initializer,
        4,
        None,
        &mut coverage_ordinal,
    )?;
    if let Some(entry) = module.entry {
        let function = function(module, entry)?;
        if !function.parameters.is_empty() {
            return Err(MlirError::UnsupportedOperation(
                "MLIR entry lowering does not yet provide process arguments".into(),
            ));
        }
        output.push_str(&format!(
            "    func.call @{}() : () -> ()\n",
            function_symbol(function)
        ));
    }
    output.push_str("    %sev_exit = arith.constant 0 : i32\n");
    output.push_str("    return %sev_exit : i32\n  }\n}\n");
    Ok(output)
}

fn render_cfg_module(module: &Module) -> Result<String, MlirError> {
    let initializer = module
        .initializer_cfg
        .as_ref()
        .expect("CFG rendering is selected only for a CFG module");
    let mut runtime_signatures = BTreeMap::<String, (Vec<LoweredType>, Option<LoweredType>)>::new();
    let mut artifact_signatures =
        BTreeMap::<ArtifactId, (Vec<LoweredType>, Vec<LoweredType>)>::new();
    let mut string_constants = BTreeMap::<ValueId, String>::new();
    let mut coverage = BTreeSet::new();
    let mut uses_task_lock = false;
    for body in std::iter::once(initializer).chain(
        module
            .functions
            .iter()
            .filter_map(|function| function.cfg.as_ref()),
    ) {
        for block in &body.blocks {
            for operation in &block.operations {
                match operation {
                    Operation::Spawn { locked: true, .. }
                    | Operation::SpawnFieldUpdate { locked: true, .. } => {
                        uses_task_lock = true;
                    }
                    Operation::RuntimeCall {
                        symbol,
                        arguments,
                        result,
                    } => {
                        let inputs = arguments
                            .iter()
                            .map(|value| value_type(module, *value))
                            .collect::<Result<Vec<_>, _>>()?;
                        let result = result.map(|value| value_type(module, value)).transpose()?;
                        runtime_signatures.insert(symbol.clone(), (inputs, result));
                    }
                    Operation::ArtifactCall {
                        artifact,
                        inputs,
                        outputs,
                    } => {
                        let inputs = inputs
                            .iter()
                            .map(|value| value_type(module, *value))
                            .collect::<Result<Vec<_>, _>>()?;
                        let outputs = outputs
                            .iter()
                            .map(|value| value_type(module, *value))
                            .collect::<Result<Vec<_>, _>>()?;
                        if let Some(known) = artifact_signatures.get(artifact) {
                            if known != &(inputs.clone(), outputs.clone()) {
                                return Err(MlirError::ArtifactSignatureConflict(*artifact));
                            }
                        }
                        artifact_signatures.insert(*artifact, (inputs, outputs));
                    }
                    Operation::Constant {
                        value: Constant::String(value),
                        result,
                    } => {
                        string_constants.insert(*result, value.clone());
                    }
                    Operation::Coverage { key } => {
                        coverage.insert(key.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    let mut assertion_messages = BTreeMap::new();
    for body in std::iter::once(initializer).chain(
        module
            .functions
            .iter()
            .filter_map(|function| function.cfg.as_ref()),
    ) {
        for block in &body.blocks {
            for operation in &block.operations {
                if let Operation::Assert {
                    condition,
                    message,
                    location,
                } = operation
                {
                    let custom = message.and_then(|message| string_constants.get(&message));
                    let failure = location.as_ref().map_or_else(
                        || custom.map_or_else(|| "assertion failed".into(), Clone::clone),
                        |location| {
                            let mut failure = format!(
                                "{}:{}:{}: assertion failed: {}",
                                location.file, location.line, location.column, location.expression
                            );
                            if let Some(custom) = custom {
                                failure.push_str(": ");
                                failure.push_str(custom);
                            }
                            failure
                        },
                    );
                    assertion_messages.insert(*condition, failure);
                }
            }
        }
    }

    let mut output = String::new();
    render_class_aliases(&mut output, module)?;
    if let Some(architecture) = &module.gpu_architecture {
        output.push_str(&format!(
            "module attributes {{severian.gpu.architecture = \"{}\"}} {{\n",
            mlir_string(architecture)
        ));
    } else {
        output.push_str("module {\n");
    }
    output.push_str("  func.func private @__sev_process_set_arguments(i32, !llvm.ptr)\n");
    output
        .push_str("  func.func private @__sev_tensor_box_unranked(i64, !llvm.ptr) -> !llvm.ptr\n");
    output.push_str("  func.func private @__sev_tensor_box_rank(!llvm.ptr) -> i64\n");
    output.push_str("  func.func private @__sev_tensor_box_descriptor(!llvm.ptr) -> !llvm.ptr\n");
    if !runtime_signatures.contains_key("__sev_tensor_local_release") {
        output.push_str("  func.func private @__sev_tensor_local_release(!llvm.ptr)\n");
    }
    if uses_task_lock {
        output.push_str("  func.func private @__sev_task_lock()\n");
        output.push_str("  func.func private @__sev_task_unlock()\n");
    }
    for global in &module.storage_globals {
        output.push_str(&format!(
            "  llvm.mlir.global internal @__sev_global_{}() : {}\n",
            global.id.0,
            mlir_type(&global.ty)?
        ));
    }
    for (value, text) in &string_constants {
        output.push_str(&format!(
            "  llvm.mlir.global private constant @{}(\"{}\\00\") : !llvm.array<{} x i8>\n",
            string_symbol(*value),
            mlir_string(text),
            text.len() + 1,
        ));
    }
    if !coverage.is_empty() {
        output.push_str("  func.func private @__sev_coverage_hit(!llvm.ptr)\n");
        for key in &coverage {
            output.push_str(&format!(
                "  llvm.mlir.global private constant @{}(\"{}\\00\") : !llvm.array<{} x i8>\n",
                coverage_symbol(key),
                mlir_string(key),
                key.len() + 1,
            ));
        }
    }
    if !assertion_messages.is_empty() {
        output.push_str("  func.func private @__sev_assert(i1, !llvm.ptr)\n");
        for (condition, message) in &assertion_messages {
            output.push_str(&format!(
                "  llvm.mlir.global private constant @__sev_assert_message_{}(\"{}\\00\") : !llvm.array<{} x i8>\n",
                condition.0,
                mlir_string(message),
                message.len() + 1,
            ));
        }
    }
    let mut declared_external_symbols = BTreeSet::new();
    let uses_aggregate_runtime = runtime_signatures.iter().any(|(symbol, (inputs, result))| {
        symbol.contains("_aggregate")
            && (inputs
                .iter()
                .any(|ty| matches!(ty, LoweredType::Aggregate(_)))
                || result
                    .as_ref()
                    .is_some_and(|ty| matches!(ty, LoweredType::Aggregate(_))))
    });
    if uses_aggregate_runtime {
        output.push_str("  func.func private @__sev_aggregate_box(!llvm.ptr, i64) -> !llvm.ptr\n");
    }
    for (symbol, (inputs, result)) in runtime_signatures {
        let aggregate_abi = symbol.contains("_aggregate");
        let inputs = inputs
            .into_iter()
            .map(|ty| runtime_abi_type(ty, aggregate_abi))
            .map(|ty| mlir_type(&ty))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let result = result
            .map(|ty| runtime_abi_type(ty, aggregate_abi))
            .map(|ty| mlir_type(&ty))
            .transpose()?
            .map(|ty| format!(" -> {ty}"))
            .unwrap_or_default();
        output.push_str(&format!(
            "  func.func private @{symbol}({inputs}){result}\n"
        ));
        declared_external_symbols.insert(symbol);
    }
    for (artifact, (inputs, outputs)) in artifact_signatures {
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(index, ty)| artifact_parameter(index, &ty))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let outputs = outputs
            .into_iter()
            .map(|ty| mlir_type(&ty))
            .collect::<Result<Vec<_>, _>>()?;
        let result = match outputs.as_slice() {
            [] => String::new(),
            [output] => format!(" -> {output}"),
            outputs => format!(" -> ({})", outputs.join(", ")),
        };
        output.push_str(&format!(
            "  func.func private @{}({inputs}){result}\n",
            artifact_symbol(artifact)
        ));
    }
    for function in module
        .functions
        .iter()
        .filter(|function| matches!(function.linkage, FunctionLinkage::External { .. }))
    {
        if declared_external_symbols.insert(function_symbol(function)) {
            render_cfg_function_declaration(&mut output, function)?;
        }
    }
    for function in module
        .functions
        .iter()
        .filter(|function| function.cfg.is_some())
    {
        render_cfg_function(&mut output, module, function)?;
    }
    render_cfg_body_function(&mut output, module, "__sev_init", &[], initializer)?;
    output.push_str("  func.func @main(%argc: i32, %argv: !llvm.ptr) -> i32 {\n");
    output.push_str(
        "    func.call @__sev_process_set_arguments(%argc, %argv) : (i32, !llvm.ptr) -> ()\n",
    );
    output.push_str("    func.call @__sev_init() : () -> ()\n");
    if let Some(entry) = module.entry {
        let function = function(module, entry)?;
        match function.parameter_types.as_slice() {
            [] => output.push_str(&format!(
                "    func.call @{}() : () -> ()\n",
                function_symbol(function)
            )),
            [LoweredType::Arguments] => {
                output.push_str(
                    "    %sev_args_undef = llvm.mlir.undef : !llvm.struct<(i32, !llvm.ptr)>\n",
                );
                output.push_str(
                    "    %sev_args_argc = llvm.insertvalue %argc, %sev_args_undef[0] : !llvm.struct<(i32, !llvm.ptr)>\n",
                );
                output.push_str(
                    "    %sev_args = llvm.insertvalue %argv, %sev_args_argc[1] : !llvm.struct<(i32, !llvm.ptr)>\n",
                );
                output.push_str(&format!(
                    "    func.call @{}(%sev_args) : (!llvm.struct<(i32, !llvm.ptr)>) -> ()\n",
                    function_symbol(function)
                ));
            }
            _ => {
                return Err(MlirError::UnsupportedOperation(
                    "entry must be `main()` or `main(arguments: args)`".into(),
                ));
            }
        }
    }
    output.push_str("    %sev_exit = arith.constant 0 : i32\n");
    output.push_str("    return %sev_exit : i32\n  }\n}\n");
    Ok(output)
}

fn render_cfg_function_declaration(
    output: &mut String,
    function: &Function,
) -> Result<(), MlirError> {
    let parameters = function
        .parameter_types
        .iter()
        .map(mlir_type)
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let result = if function.result == LoweredType::Unit {
        String::new()
    } else {
        format!(" -> {}", mlir_type(&function.result)?)
    };
    output.push_str(&format!(
        "  func.func private @{}({parameters}){result}\n",
        function_symbol(function)
    ));
    Ok(())
}

fn render_cfg_function(
    output: &mut String,
    module: &Module,
    function: &Function,
) -> Result<(), MlirError> {
    render_cfg_body_function(
        output,
        module,
        &function_symbol(function),
        &function.parameter_types,
        function.cfg.as_ref().expect("filtered CFG function"),
    )
}

fn render_cfg_body_function(
    output: &mut String,
    module: &Module,
    symbol: &str,
    parameter_types: &[LoweredType],
    body: &severian_lir::CfgBody,
) -> Result<(), MlirError> {
    let parameters = parameter_types
        .iter()
        .enumerate()
        .map(|(index, ty)| Ok(format!("%arg{index}: {}", mlir_type(ty)?)))
        .collect::<Result<Vec<_>, MlirError>>()?
        .join(", ");
    let result = if body.return_type == LoweredType::Unit {
        String::new()
    } else {
        format!(" -> {}", mlir_type(&body.return_type)?)
    };
    output.push_str(&format!("  func.func @{symbol}({parameters}){result} {{\n"));
    let mut ssa_locals = BTreeMap::new();
    let ssa_live_ins = cfg_ssa_live_ins(body);
    let gpu_regions = gpu_regions(body)?;
    let gpu_blocks = gpu_regions
        .values()
        .flat_map(|region| region.blocks.iter().copied())
        .collect::<BTreeSet<_>>();
    for block in &body.blocks {
        if gpu_blocks.contains(&block.id) {
            continue;
        }
        if block.id != body.entry {
            ssa_locals.clear();
            let arguments = ssa_live_ins
                .get(&block.id)
                .into_iter()
                .flatten()
                .map(|local| {
                    let name = format!("%local{}_bb{}", local.0, block.id.0);
                    ssa_locals.insert(*local, name.clone());
                    let ty = &body.locals[local.0 as usize].ty;
                    Ok(format!("{name}: {}", mlir_type(ty)?))
                })
                .collect::<Result<Vec<_>, MlirError>>()?
                .join(", ");
            output.push_str(&format!("  ^bb{}({arguments}):\n", block.id.0));
        }
        if block.id == body.entry {
            output.push_str("    %sev_one = arith.constant 1 : i64\n");
            for global in &module.storage_globals {
                output.push_str(&format!(
                    "    %global{}_address = llvm.mlir.addressof @__sev_global_{} : !llvm.ptr\n",
                    global.id.0, global.id.0
                ));
            }
            for local in &body.locals {
                if is_ssa_local_type(&local.ty) {
                    continue;
                }
                output.push_str(&format!(
                    "    %local{} = llvm.alloca %sev_one x {} : (i64) -> !llvm.ptr\n",
                    local.id.0,
                    cfg_local_storage_type(&local.ty)?
                ));
            }
            if body
                .locals
                .iter()
                .any(|local| !local.argument && matches!(local.ty, LoweredType::Tensor { .. }))
            {
                output.push_str("    %sev_null_tensor_box = llvm.mlir.zero : !llvm.ptr\n");
                for local in body.locals.iter().filter(|local| {
                    !local.argument && matches!(local.ty, LoweredType::Tensor { .. })
                }) {
                    output.push_str(&format!(
                        "    llvm.store %sev_null_tensor_box, %local{} : !llvm.ptr, !llvm.ptr\n",
                        local.id.0
                    ));
                }
            }
            for (argument, local) in body
                .locals
                .iter()
                .filter(|local| local.argument)
                .enumerate()
            {
                if is_ssa_local_type(&local.ty) {
                    ssa_locals.insert(local.id, format!("%arg{argument}"));
                    continue;
                }
                if matches!(local.ty, LoweredType::Tensor { .. }) {
                    let boxed = format!("%argument_tensor_box_{argument}");
                    render_tensor_aggregate_box(
                        output,
                        &format!("%arg{argument}"),
                        &local.ty,
                        &local.ty,
                        &boxed,
                        &format!("argument_tensor_{argument}"),
                        4,
                    )?;
                    output.push_str(&format!(
                        "    llvm.store {boxed}, %local{} : !llvm.ptr, !llvm.ptr\n",
                        local.id.0
                    ));
                    continue;
                }
                output.push_str(&format!(
                    "    llvm.store %arg{argument}, %local{} : {}, !llvm.ptr\n",
                    local.id.0,
                    mlir_type(&local.ty)?
                ));
            }
        }
        for (operation_index, operation) in block.operations.iter().enumerate() {
            render_cfg_operation(
                output,
                module,
                body,
                block.id,
                operation_index,
                operation,
                4,
                &mut ssa_locals,
            )?;
        }
        if let severian_lir::Terminator::Goto(target) = block.terminator {
            if let Some(region) = gpu_regions.get(&target) {
                render_gpu_region(output, module, body, region, &mut ssa_locals)?;
                output.push_str(&format!(
                    "    cf.br {}\n",
                    cfg_branch_target(region.exit, &ssa_live_ins, &ssa_locals, body)?
                ));
                continue;
            }
        }
        render_cfg_terminator(
            output,
            module,
            body,
            &block.terminator,
            4,
            &mut ssa_locals,
            &ssa_live_ins,
        )?;
    }
    output.push_str("  }\n");
    Ok(())
}

fn is_ssa_local_type(ty: &LoweredType) -> bool {
    matches!(ty, LoweredType::Task(_))
}

fn cfg_local_storage_type(ty: &LoweredType) -> Result<String, MlirError> {
    if matches!(ty, LoweredType::Tensor { .. }) {
        Ok("!llvm.ptr".into())
    } else {
        mlir_type(ty)
    }
}

fn cfg_ssa_live_ins(
    body: &severian_lir::CfgBody,
) -> BTreeMap<severian_lir::BlockId, Vec<severian_lir::LocalId>> {
    let mut uses = BTreeMap::new();
    let mut definitions = BTreeMap::new();
    for block in &body.blocks {
        let mut used = BTreeSet::new();
        let mut defined = BTreeSet::new();
        for operation in &block.operations {
            match operation {
                Operation::Load { place, .. } => {
                    if let severian_lir::PlaceBase::Local(local) = place.base {
                        if place.projection.is_empty()
                            && is_ssa_local_type(&body.locals[local.0 as usize].ty)
                            && !defined.contains(&local)
                        {
                            used.insert(local);
                        }
                    }
                }
                Operation::Store { place, .. } => {
                    if let severian_lir::PlaceBase::Local(local) = place.base {
                        if place.projection.is_empty()
                            && is_ssa_local_type(&body.locals[local.0 as usize].ty)
                        {
                            defined.insert(local);
                        }
                    }
                }
                _ => {}
            }
        }
        if let severian_lir::Terminator::Call {
            destination: Some(destination),
            ..
        } = &block.terminator
        {
            if let severian_lir::PlaceBase::Local(local) = destination.base {
                if destination.projection.is_empty()
                    && is_ssa_local_type(&body.locals[local.0 as usize].ty)
                {
                    defined.insert(local);
                }
            }
        }
        uses.insert(block.id, used);
        definitions.insert(block.id, defined);
    }
    let mut live_in = body
        .blocks
        .iter()
        .map(|block| (block.id, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in body.blocks.iter().rev() {
            let mut next = uses[&block.id].clone();
            for successor in cfg_successors(&block.terminator) {
                for local in &live_in[&successor] {
                    if !definitions[&block.id].contains(local) {
                        next.insert(*local);
                    }
                }
            }
            if next != live_in[&block.id] {
                live_in.insert(block.id, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    live_in
        .into_iter()
        .map(|(block, locals)| (block, locals.into_iter().collect()))
        .collect()
}

fn cfg_branch_target(
    target: severian_lir::BlockId,
    live_ins: &BTreeMap<severian_lir::BlockId, Vec<severian_lir::LocalId>>,
    values: &BTreeMap<severian_lir::LocalId, String>,
    body: &severian_lir::CfgBody,
) -> Result<String, MlirError> {
    let arguments = live_ins
        .get(&target)
        .into_iter()
        .flatten()
        .map(|local| {
            let value = values.get(local).ok_or_else(|| {
                MlirError::UnsupportedOperation(format!(
                    "SSA local {} reaches block {} without a value",
                    local.0, target.0
                ))
            })?;
            Ok((value.clone(), mlir_type(&body.locals[local.0 as usize].ty)?))
        })
        .collect::<Result<Vec<_>, MlirError>>()?;
    if arguments.is_empty() {
        Ok(format!("^bb{}", target.0))
    } else {
        let values = arguments
            .iter()
            .map(|(value, _)| value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = arguments
            .iter()
            .map(|(_, ty)| ty.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!("^bb{}({values} : {types})", target.0))
    }
}

#[derive(Debug)]
struct GpuCfgRegion {
    entry: severian_lir::BlockId,
    exit: severian_lir::BlockId,
    blocks: BTreeSet<severian_lir::BlockId>,
}

fn cfg_successors(terminator: &severian_lir::Terminator) -> Vec<severian_lir::BlockId> {
    match terminator {
        severian_lir::Terminator::Goto(target) => vec![*target],
        severian_lir::Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        severian_lir::Terminator::Switch {
            targets, fallback, ..
        } => targets
            .iter()
            .map(|(_, target)| *target)
            .chain(std::iter::once(*fallback))
            .collect(),
        severian_lir::Terminator::Call { target, .. } => vec![*target],
        severian_lir::Terminator::Return(_)
        | severian_lir::Terminator::Throw(_)
        | severian_lir::Terminator::Unreachable => Vec::new(),
    }
}

/// Finds maximal CFG components selected for GPU execution. Placement is
/// deliberately analyzed after ordinary lowering, so this is independent of
/// the source operator or the value type being processed.
fn gpu_regions(
    body: &severian_lir::CfgBody,
) -> Result<BTreeMap<severian_lir::BlockId, GpuCfgRegion>, MlirError> {
    use severian_universal::ExecutionPlacement;

    let gpu = body
        .blocks
        .iter()
        .filter(|block| block.execution == Some(ExecutionPlacement::Gpu))
        .map(|block| block.id)
        .collect::<BTreeSet<_>>();
    let entries = gpu
        .iter()
        .copied()
        .filter(|candidate| {
            body.blocks.iter().any(|block| {
                !gpu.contains(&block.id) && cfg_successors(&block.terminator).contains(candidate)
            })
        })
        .collect::<Vec<_>>();
    let mut regions = BTreeMap::new();
    let mut assigned = BTreeSet::new();
    for entry in entries {
        let mut pending = vec![entry];
        let mut blocks = BTreeSet::new();
        let mut exits = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !blocks.insert(id) {
                continue;
            }
            let block = body.blocks.get(id.0 as usize).ok_or_else(|| {
                MlirError::UnsupportedOperation(format!("unknown GPU block {}", id.0))
            })?;
            for successor in cfg_successors(&block.terminator) {
                if gpu.contains(&successor) {
                    pending.push(successor);
                } else {
                    exits.insert(successor);
                }
            }
        }
        if exits.len() != 1 {
            return Err(MlirError::UnsupportedOperation(format!(
                "GPU CFG region at block {} requires one host continuation, found {}",
                entry.0,
                exits.len()
            )));
        }
        if blocks.iter().any(|block| !assigned.insert(*block)) {
            return Err(MlirError::UnsupportedOperation(
                "overlapping GPU CFG regions are not supported".into(),
            ));
        }
        regions.insert(
            entry,
            GpuCfgRegion {
                entry,
                exit: *exits.iter().next().expect("checked one exit"),
                blocks,
            },
        );
    }
    if assigned != gpu {
        return Err(MlirError::UnsupportedOperation(
            "GPU CFG region has no ordinary host entry".into(),
        ));
    }
    Ok(regions)
}

fn render_gpu_region(
    output: &mut String,
    module: &Module,
    body: &severian_lir::CfgBody,
    region: &GpuCfgRegion,
    ssa_locals: &mut BTreeMap<severian_lir::LocalId, String>,
) -> Result<(), MlirError> {
    output.push_str(&format!(
        "    %gpu_one_{} = arith.constant 1 : index\n",
        region.entry.0
    ));
    output.push_str(&format!(
        "    gpu.launch blocks(%gpu_bx_{0}, %gpu_by_{0}, %gpu_bz_{0}) in (%gpu_gx_{0} = %gpu_one_{0}, %gpu_gy_{0} = %gpu_one_{0}, %gpu_gz_{0} = %gpu_one_{0}) threads(%gpu_tx_{0}, %gpu_ty_{0}, %gpu_tz_{0}) in (%gpu_sx_{0} = %gpu_one_{0}, %gpu_sy_{0} = %gpu_one_{0}, %gpu_sz_{0} = %gpu_one_{0}) {{\n",
        region.entry.0
    ));
    for id in &region.blocks {
        let block = body.blocks.get(id.0 as usize).ok_or_else(|| {
            MlirError::UnsupportedOperation(format!("unknown GPU block {}", id.0))
        })?;
        if *id != region.entry {
            output.push_str(&format!("    ^gpu{}_bb{}:\n", region.entry.0, id.0));
        }
        for (operation_index, operation) in block.operations.iter().enumerate() {
            render_cfg_operation(
                output,
                module,
                body,
                block.id,
                operation_index,
                operation,
                6,
                ssa_locals,
            )?;
        }
        render_gpu_terminator(output, module, &block.terminator, region, 6)?;
    }
    output.push_str(&format!("    ^gpu{}_exit:\n", region.entry.0));
    output.push_str("      gpu.terminator\n");
    output.push_str("    }\n");
    Ok(())
}

fn gpu_label(target: severian_lir::BlockId, region: &GpuCfgRegion) -> Result<String, MlirError> {
    if target == region.exit {
        Ok(format!("^gpu{}_exit", region.entry.0))
    } else if region.blocks.contains(&target) {
        Ok(format!("^gpu{}_bb{}", region.entry.0, target.0))
    } else {
        Err(MlirError::UnsupportedOperation(format!(
            "GPU CFG branch escapes to block {} instead of continuation {}",
            target.0, region.exit.0
        )))
    }
}

fn render_gpu_terminator(
    output: &mut String,
    module: &Module,
    terminator: &severian_lir::Terminator,
    region: &GpuCfgRegion,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    match terminator {
        severian_lir::Terminator::Goto(target) => {
            output.push_str(&format!(
                "{indentation}cf.br {}\n",
                gpu_label(*target, region)?
            ));
        }
        severian_lir::Terminator::Branch {
            condition,
            then_block,
            else_block,
        } => output.push_str(&format!(
            "{indentation}cf.cond_br %v{}, {}, {}\n",
            condition.0,
            gpu_label(*then_block, region)?,
            gpu_label(*else_block, region)?
        )),
        severian_lir::Terminator::Switch { .. } => {
            return Err(MlirError::UnsupportedOperation(
                "switch terminators in GPU CFG regions are not implemented".into(),
            ));
        }
        severian_lir::Terminator::Call {
            function: callee,
            arguments,
            destination,
            target,
        } => {
            let callee = function(module, *callee)?;
            if arguments.len() != callee.parameter_types.len() {
                return Err(MlirError::UnsupportedOperation(format!(
                    "call to {} has {} arguments for {} parameters",
                    function_symbol(callee),
                    arguments.len(),
                    callee.parameter_types.len()
                )));
            }
            let mut lowered_arguments = Vec::with_capacity(arguments.len());
            for (index, (value, target_type)) in
                arguments.iter().zip(&callee.parameter_types).enumerate()
            {
                let source_type = value_type(module, *value)?;
                let source = format!("%v{}", value.0);
                let result = format!("%call_arg_{}_{}", target.0, index);
                let lowered = if source_type == *target_type {
                    source
                } else if matches!(source_type, LoweredType::Tensor { .. })
                    && matches!(target_type, LoweredType::Tensor { .. })
                {
                    output.push_str(&format!(
                        "{indentation}{result} = tensor.cast {source} : {} to {}\n",
                        mlir_type(&source_type)?,
                        mlir_type(target_type)?
                    ));
                    result
                } else {
                    render_named_conversion(
                        output,
                        &source,
                        &source_type,
                        target_type,
                        &result,
                        indent,
                    )?
                };
                lowered_arguments.push(lowered);
            }
            let argument_values = lowered_arguments.join(", ");
            let argument_types = callee
                .parameter_types
                .iter()
                .map(mlir_type)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let result_type = mlir_type(&callee.result)?;
            if callee.result == LoweredType::Unit {
                output.push_str(&format!(
                    "{indentation}func.call @{}({argument_values}) : ({argument_types}) -> ()\n",
                    function_symbol(callee)
                ));
            } else if let Some(destination) = destination {
                output.push_str(&format!(
                    "{indentation}%gpu_call_result_{}_{} = func.call @{}({argument_values}) : ({argument_types}) -> {result_type}\n",
                    region.entry.0,
                    target.0,
                    function_symbol(callee)
                ));
                output.push_str(&format!(
                    "{indentation}llvm.store %gpu_call_result_{}_{}, {} : {result_type}, !llvm.ptr\n",
                    region.entry.0,
                    target.0,
                    cfg_place_address(destination)?
                ));
            } else {
                output.push_str(&format!(
                    "{indentation}func.call @{}({argument_values}) : ({argument_types}) -> {result_type}\n",
                    function_symbol(callee)
                ));
            }
            output.push_str(&format!(
                "{indentation}cf.br {}\n",
                gpu_label(*target, region)?
            ));
        }
        severian_lir::Terminator::Return(_)
        | severian_lir::Terminator::Throw(_)
        | severian_lir::Terminator::Unreachable => {
            return Err(MlirError::UnsupportedOperation(
                "GPU CFG regions must return control to their host continuation".into(),
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_cfg_operation(
    output: &mut String,
    module: &Module,
    body: &severian_lir::CfgBody,
    block: severian_lir::BlockId,
    operation_index: usize,
    operation: &Operation,
    indent: usize,
    ssa_locals: &mut BTreeMap<severian_lir::LocalId, String>,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    match operation {
        Operation::Coverage { key } => {
            let symbol = coverage_symbol(key);
            let value = format!("{symbol}_bb{}_op{operation_index}", block.0);
            output.push_str(&format!(
                "{indentation}%{value} = llvm.mlir.addressof @{symbol} : !llvm.ptr\n"
            ));
            output.push_str(&format!(
                "{indentation}func.call @__sev_coverage_hit(%{value}) : (!llvm.ptr) -> ()\n"
            ));
        }
        Operation::Constant { value, result } => {
            let ty = value_type(module, *result)?;
            if matches!(value, Constant::String(_)) {
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.mlir.addressof @{} : !llvm.ptr\n",
                    result.0,
                    string_symbol(*result)
                ));
            } else if matches!(value, Constant::None) && mlir_type(&ty)? == "!llvm.ptr" {
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.mlir.zero : !llvm.ptr\n",
                    result.0
                ));
            } else {
                let literal = match value {
                    Constant::Integer(value) => value.clone(),
                    Constant::Float(value) => mlir_float_literal(value),
                    Constant::Boolean(true) => "1".into(),
                    Constant::Boolean(false) | Constant::None | Constant::Unit => "0".into(),
                    Constant::Bytes(_) | Constant::String(_) => unreachable!(),
                };
                output.push_str(&format!(
                    "{indentation}%v{} = arith.constant {literal} : {}\n",
                    result.0,
                    mlir_type(&ty)?
                ));
            }
        }
        Operation::Aggregate {
            class: _,
            fields,
            result,
        } => {
            let aggregate_type = value_type(module, *result)?;
            let ty = mlir_type(&aggregate_type)?;
            if fields.is_empty() {
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.mlir.undef : {ty}\n",
                    result.0
                ));
            } else {
                output.push_str(&format!(
                    "{indentation}%aggregate_{}_0 = llvm.mlir.undef : {ty}\n",
                    result.0
                ));
                for (index, field) in fields.iter().enumerate() {
                    let input = format!("%aggregate_{}_{}", result.0, index);
                    let output_value = if index + 1 == fields.len() {
                        format!("%v{}", result.0)
                    } else {
                        format!("%aggregate_{}_{}", result.0, index + 1)
                    };
                    let field_type = value_type(module, *field)?;
                    let target_type =
                        aggregate_declaration_field_type(module, &aggregate_type, index)?;
                    let field_value = if tensor_aggregate_abi_type(&target_type).is_some() {
                        let converted = format!("%aggregate_field_{}_{}", result.0, index);
                        render_tensor_aggregate_box(
                            output,
                            &format!("%v{}", field.0),
                            &field_type,
                            target_type,
                            &converted,
                            &format!("aggregate_{}_{}", result.0, index),
                            indent,
                        )?;
                        converted
                    } else {
                        render_named_conversion(
                            output,
                            &format!("%v{}", field.0),
                            &field_type,
                            &target_type,
                            &format!("%aggregate_scalar_{}_{}", result.0, index),
                            indent,
                        )?
                    };
                    output.push_str(&format!(
                        "{indentation}{output_value} = llvm.insertvalue {field_value}, {input}[{index}] : {ty}\n"
                    ));
                }
            }
        }
        Operation::Load { place, result } => {
            let ty = value_type(module, *result)?;
            if let severian_lir::PlaceBase::Local(local) = place.base {
                if place.projection.is_empty()
                    && body
                        .locals
                        .get(local.0 as usize)
                        .is_some_and(|local| matches!(local.ty, LoweredType::Tensor { .. }))
                {
                    let local_type = &body.locals[local.0 as usize].ty;
                    let boxed = format!("%load_tensor_box_b{}_o{}", block.0, operation_index);
                    output.push_str(&format!(
                        "{indentation}{boxed} = llvm.load %local{} : !llvm.ptr -> !llvm.ptr\n",
                        local.0
                    ));
                    render_tensor_aggregate_unbox(
                        output,
                        &boxed,
                        local_type,
                        &ty,
                        &format!("%v{}", result.0),
                        &format!("load_tensor_local_b{}_o{}", block.0, operation_index),
                        indent,
                    )?;
                    return Ok(());
                }
                if place.projection.is_empty()
                    && body
                        .locals
                        .get(local.0 as usize)
                        .is_some_and(|local| is_ssa_local_type(&local.ty))
                {
                    let source = ssa_locals.get(&local).cloned().ok_or_else(|| {
                        MlirError::UnsupportedOperation(format!(
                            "SSA local {} ({ty:?}) is loaded before its value is stored in block {}",
                            local.0,
                            block.0
                        ))
                    })?;
                    let spelling = mlir_type(&ty)?;
                    output.push_str(&format!(
                        "{indentation}%v{} = builtin.unrealized_conversion_cast {source} : {spelling} to {spelling}\n",
                        result.0
                    ));
                    return Ok(());
                }
            }
            if let [severian_lir::Projection::Field(field)] = place.projection.as_slice() {
                let base_type = mlir_type(&cfg_place_base_type(module, body, place)?)?;
                let stored_type = cfg_aggregate_field_type(module, body, place, *field)?;
                output.push_str(&format!(
                    "{indentation}%load_base_b{}_o{} = llvm.load {} : !llvm.ptr -> {base_type}\n",
                    block.0,
                    operation_index,
                    cfg_place_base_address(place)
                ));
                if tensor_aggregate_abi_type(&stored_type).is_some() {
                    output.push_str(&format!(
                        "{indentation}%load_tensor_b{}_o{} = llvm.extractvalue %load_base_b{}_o{}[{}] : {base_type}\n",
                        block.0, operation_index, block.0, operation_index, field
                    ));
                    render_tensor_aggregate_unbox(
                        output,
                        &format!("%load_tensor_b{}_o{}", block.0, operation_index),
                        stored_type,
                        &ty,
                        &format!("%v{}", result.0),
                        &format!("load_tensor_b{}_o{}", block.0, operation_index),
                        indent,
                    )?;
                } else {
                    let extracted = format!("%load_field_b{}_o{}", block.0, operation_index);
                    output.push_str(&format!(
                        "{indentation}{extracted} = llvm.extractvalue %load_base_b{}_o{}[{}] : {base_type}\n",
                        block.0, operation_index, field
                    ));
                    let converted = render_named_conversion(
                        output,
                        &extracted,
                        &stored_type,
                        &ty,
                        &format!("%v{}", result.0),
                        indent,
                    )?;
                    if converted == extracted {
                        output.push_str(&format!(
                            "{indentation}%v{} = builtin.unrealized_conversion_cast {extracted} : {} to {}\n",
                            result.0,
                            mlir_type(&ty)?,
                            mlir_type(&ty)?
                        ));
                    }
                }
            } else {
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.load {} : !llvm.ptr -> {}\n",
                    result.0,
                    cfg_place_address(place)?,
                    mlir_type(&ty)?
                ));
            }
        }
        Operation::AddressOf { place, result } => {
            output.push_str(&format!(
                "{indentation}%v{} = builtin.unrealized_conversion_cast {} : !llvm.ptr to !llvm.ptr\n",
                result.0,
                cfg_place_address(place)?
            ));
        }
        Operation::Store { place, value } => {
            let ty = value_type(module, *value)?;
            if let severian_lir::PlaceBase::Local(local) = place.base {
                if place.projection.is_empty()
                    && body
                        .locals
                        .get(local.0 as usize)
                        .is_some_and(|local| matches!(local.ty, LoweredType::Tensor { .. }))
                {
                    let target_type = &body.locals[local.0 as usize].ty;
                    let boxed = format!("%store_tensor_box_b{}_o{}", block.0, operation_index);
                    render_tensor_aggregate_box(
                        output,
                        &format!("%v{}", value.0),
                        &ty,
                        target_type,
                        &boxed,
                        &format!("store_tensor_local_b{}_o{}", block.0, operation_index),
                        indent,
                    )?;
                    output.push_str(&format!(
                        "{indentation}func.call @__sev_tensor_local_release(%local{}) : (!llvm.ptr) -> ()\n",
                        local.0
                    ));
                    output.push_str(&format!(
                        "{indentation}llvm.store {boxed}, %local{} : !llvm.ptr, !llvm.ptr\n",
                        local.0
                    ));
                    return Ok(());
                }
                if place.projection.is_empty()
                    && body
                        .locals
                        .get(local.0 as usize)
                        .is_some_and(|local| is_ssa_local_type(&local.ty))
                {
                    let target_type = &body.locals[local.0 as usize].ty;
                    let source = format!("%v{}", value.0);
                    let converted = format!("%ssa_store_b{}_o{}", block.0, operation_index);
                    let stored = if ty == *target_type {
                        source
                    } else if matches!(ty, LoweredType::Tensor { .. })
                        && matches!(target_type, LoweredType::Tensor { .. })
                    {
                        output.push_str(&format!(
                            "{indentation}{converted} = tensor.cast {source} : {} to {}\n",
                            mlir_type(&ty)?,
                            mlir_type(target_type)?
                        ));
                        converted
                    } else {
                        render_named_conversion(
                            output,
                            &source,
                            &ty,
                            target_type,
                            &converted,
                            indent,
                        )?
                    };
                    ssa_locals.insert(local, stored);
                    return Ok(());
                }
            }
            if let [severian_lir::Projection::Field(field)] = place.projection.as_slice() {
                let base_type = mlir_type(&cfg_place_base_type(module, body, place)?)?;
                let address = cfg_place_base_address(place);
                let stored_type = cfg_aggregate_field_type(module, body, place, *field)?;
                output.push_str(&format!(
                    "{indentation}%store_base_b{}_o{} = llvm.load {address} : !llvm.ptr -> {base_type}\n",
                    block.0, operation_index
                ));
                let stored = if tensor_aggregate_abi_type(&stored_type).is_some() {
                    render_tensor_aggregate_box(
                        output,
                        &format!("%v{}", value.0),
                        &ty,
                        stored_type,
                        &format!("%store_tensor_b{}_o{}", block.0, operation_index),
                        &format!("store_tensor_b{}_o{}", block.0, operation_index),
                        indent,
                    )?;
                    format!("%store_tensor_b{}_o{}", block.0, operation_index)
                } else {
                    render_named_conversion(
                        output,
                        &format!("%v{}", value.0),
                        &ty,
                        &stored_type,
                        &format!("%store_scalar_b{}_o{}", block.0, operation_index),
                        indent,
                    )?
                };
                output.push_str(&format!(
                    "{indentation}%store_value_b{}_o{} = llvm.insertvalue {stored}, %store_base_b{}_o{}[{}] : {base_type}\n",
                    block.0, operation_index, block.0, operation_index, field
                ));
                output.push_str(&format!(
                    "{indentation}llvm.store %store_value_b{}_o{}, {address} : {base_type}, !llvm.ptr\n",
                    block.0, operation_index
                ));
            } else {
                output.push_str(&format!(
                    "{indentation}llvm.store %v{}, {} : {}, !llvm.ptr\n",
                    value.0,
                    cfg_place_address(place)?,
                    mlir_type(&ty)?
                ));
            }
        }
        Operation::Unary {
            operator,
            operand,
            result,
        } => render_cfg_unary(output, module, *operator, *operand, *result, indent)?,
        Operation::Convert {
            operand, result, ..
        } => render_conversion(output, module, *operand, *result, indent)?,
        Operation::Binary {
            operator,
            left,
            right,
            result,
        } => {
            let input = value_type(module, *left)?;
            output.push_str(&format!(
                "{indentation}%v{} = {} %v{}, %v{} : {}\n",
                result.0,
                binary_mnemonic(*operator, &input)?,
                left.0,
                right.0,
                mlir_type(&input)?
            ));
        }
        Operation::RuntimeCall {
            symbol,
            arguments,
            result,
        } => render_runtime_call(output, module, symbol, arguments, *result, indent)?,
        Operation::ArtifactCall {
            artifact,
            inputs,
            outputs,
        } => {
            let input_values = inputs
                .iter()
                .map(|value| format!("%v{}", value.0))
                .collect::<Vec<_>>()
                .join(", ");
            let input_types = inputs
                .iter()
                .map(|value| mlir_type(&value_type(module, *value)?))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let output_values = outputs
                .iter()
                .map(|value| format!("%v{}", value.0))
                .collect::<Vec<_>>()
                .join(", ");
            let output_types = outputs
                .iter()
                .map(|value| mlir_type(&value_type(module, *value)?))
                .collect::<Result<Vec<_>, _>>()?;
            let result = match output_types.as_slice() {
                [] => String::new(),
                [output] => format!(" -> {output}"),
                outputs => format!(" -> ({})", outputs.join(", ")),
            };
            let assignment = (!output_values.is_empty())
                .then(|| format!("{output_values} = "))
                .unwrap_or_default();
            output.push_str(&format!(
                "{indentation}{assignment}func.call @{}({input_values}) : ({input_types}){result}\n",
                artifact_symbol(*artifact)
            ));
        }
        Operation::Spawn {
            function: target,
            arguments,
            result,
            owner,
            locked,
        } => {
            let target = function(module, *target)?;
            let arguments_text = arguments
                .iter()
                .map(|value| format!("%v{}", value.0))
                .collect::<Vec<_>>()
                .join(", ");
            let argument_types = arguments
                .iter()
                .map(|value| mlir_type(&value_type(module, *value)?))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let owner = match owner {
                severian_lir::TaskOwner::SelfScope => "self",
                severian_lir::TaskOwner::Runtime => "runtime",
                severian_lir::TaskOwner::Inferred => "inferred",
            };
            let attributes =
                format!("attributes {{severian.owner = \"{owner}\", severian.locked = {locked}}}");
            if target.result == LoweredType::Unit {
                output.push_str(&format!(
                    "{indentation}%v{} = async.execute {attributes} {{\n",
                    result.0
                ));
                if *locked {
                    output.push_str(&format!(
                        "{}func.call @__sev_task_lock() : () -> ()\n",
                        " ".repeat(indent + 2)
                    ));
                }
                output.push_str(&format!(
                    "{}func.call @{}({arguments_text}) : ({argument_types}) -> ()\n",
                    " ".repeat(indent + 2),
                    function_symbol(target)
                ));
                if *locked {
                    output.push_str(&format!(
                        "{}func.call @__sev_task_unlock() : () -> ()\n",
                        " ".repeat(indent + 2)
                    ));
                }
                output.push_str(&format!("{}async.yield\n", " ".repeat(indent + 2)));
            } else {
                let result_type = mlir_type(&target.result)?;
                output.push_str(&format!(
                    "{indentation}%task_token{}, %v{} = async.execute -> !async.value<{result_type}> {attributes} {{\n",
                    result.0, result.0
                ));
                if *locked {
                    output.push_str(&format!(
                        "{}func.call @__sev_task_lock() : () -> ()\n",
                        " ".repeat(indent + 2)
                    ));
                }
                output.push_str(&format!(
                    "{}%task_value{} = func.call @{}({arguments_text}) : ({argument_types}) -> {result_type}\n",
                    " ".repeat(indent + 2),
                    result.0,
                    function_symbol(target)
                ));
                if *locked {
                    output.push_str(&format!(
                        "{}func.call @__sev_task_unlock() : () -> ()\n",
                        " ".repeat(indent + 2)
                    ));
                }
                output.push_str(&format!(
                    "{}async.yield %task_value{} : {result_type}\n",
                    " ".repeat(indent + 2),
                    result.0
                ));
            }
            output.push_str(&format!("{indentation}}}\n"));
        }
        Operation::SpawnFieldUpdate {
            place,
            operator,
            value,
            result,
            owner,
            locked,
        } => {
            let [severian_lir::Projection::Field(field)] = place.projection.as_slice() else {
                return Err(MlirError::UnsupportedOperation(format!(
                    "async update requires one field projection, got {place:?}"
                )));
            };
            let owner = match owner {
                severian_lir::TaskOwner::SelfScope => "self",
                severian_lir::TaskOwner::Runtime => "runtime",
                severian_lir::TaskOwner::Inferred => "inferred",
            };
            let attributes =
                format!("attributes {{severian.owner = \"{owner}\", severian.locked = {locked}}}");
            let nested = " ".repeat(indent + 2);
            let base_type = mlir_type(&cfg_place_base_type(module, body, place)?)?;
            let field_type = value_type(module, *value)?;
            let field_spelling = mlir_type(&field_type)?;
            let address = cfg_place_base_address(place);
            output.push_str(&format!(
                "{indentation}%v{} = async.execute {attributes} {{\n",
                result.0
            ));
            if *locked {
                output.push_str(&format!(
                    "{nested}func.call @__sev_task_lock() : () -> ()\n"
                ));
            }
            output.push_str(&format!(
                "{nested}%update_base_{} = llvm.load {address} : !llvm.ptr -> {base_type}\n",
                result.0
            ));
            output.push_str(&format!(
                "{nested}%update_old_{} = llvm.extractvalue %update_base_{}[{}] : {base_type}\n",
                result.0, result.0, field
            ));
            output.push_str(&format!(
                "{nested}%update_new_{} = {} %update_old_{}, %v{} : {field_spelling}\n",
                result.0,
                binary_mnemonic(*operator, &field_type)?,
                result.0,
                value.0
            ));
            output.push_str(&format!(
                "{nested}%update_result_{} = llvm.insertvalue %update_new_{}, %update_base_{}[{}] : {base_type}\n",
                result.0, result.0, result.0, field
            ));
            output.push_str(&format!(
                "{nested}llvm.store %update_result_{}, {address} : {base_type}, !llvm.ptr\n",
                result.0
            ));
            if *locked {
                output.push_str(&format!(
                    "{nested}func.call @__sev_task_unlock() : () -> ()\n"
                ));
            }
            output.push_str(&format!("{nested}async.yield\n"));
            output.push_str(&format!("{indentation}}}\n"));
        }
        Operation::Await { task, result } => {
            let ty = value_type(module, *result)?;
            if ty == LoweredType::Unit {
                output.push_str(&format!(
                    "{indentation}async.await %v{} : !async.token\n",
                    task.0
                ));
                output.push_str(&format!(
                    "{indentation}%v{} = arith.constant 0 : i8\n",
                    result.0
                ));
            } else {
                let result_type = value_type(module, *task)?.task_result().ok_or_else(|| {
                    MlirError::UnsupportedOperation("await operand is not a task".into())
                })?;
                output.push_str(&format!(
                    "{indentation}%v{} = async.await %v{} : !async.value<{}>\n",
                    result.0,
                    task.0,
                    mlir_type(&result_type)?
                ));
            }
        }
        Operation::Assert {
            condition,
            message,
            location,
        } => render_assert(
            output,
            module,
            *condition,
            *message,
            location.as_ref(),
            indent,
        )?,
        unsupported => {
            return Err(MlirError::UnsupportedOperation(format!(
                "CFG operation {unsupported:?}"
            )));
        }
    }
    Ok(())
}

fn render_cfg_terminator(
    output: &mut String,
    module: &Module,
    body: &severian_lir::CfgBody,
    terminator: &severian_lir::Terminator,
    indent: usize,
    ssa_locals: &mut BTreeMap<severian_lir::LocalId, String>,
    ssa_live_ins: &BTreeMap<severian_lir::BlockId, Vec<severian_lir::LocalId>>,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    match terminator {
        severian_lir::Terminator::Goto(target) => {
            output.push_str(&format!(
                "{indentation}cf.br {}\n",
                cfg_branch_target(*target, ssa_live_ins, ssa_locals, body)?
            ));
        }
        severian_lir::Terminator::Branch {
            condition,
            then_block,
            else_block,
        } => output.push_str(&format!(
            "{indentation}cf.cond_br %v{}, {}, {}\n",
            condition.0,
            cfg_branch_target(*then_block, ssa_live_ins, ssa_locals, body)?,
            cfg_branch_target(*else_block, ssa_live_ins, ssa_locals, body)?
        )),
        severian_lir::Terminator::Switch {
            discriminant,
            targets,
            fallback,
        } => {
            let targets = targets
                .iter()
                .map(|(case, target)| {
                    let value = match case {
                        severian_lir::Case::Integer(value) => value.to_string(),
                        severian_lir::Case::Boolean(value) => u8::from(*value).to_string(),
                        severian_lir::Case::Variant(value) => value.to_string(),
                    };
                    Ok(format!(
                        "{value}: {}",
                        cfg_branch_target(*target, ssa_live_ins, ssa_locals, body)?
                    ))
                })
                .collect::<Result<Vec<_>, MlirError>>()?
                .join(", ");
            let targets = if targets.is_empty() {
                format!(
                    "default: {}",
                    cfg_branch_target(*fallback, ssa_live_ins, ssa_locals, body)?
                )
            } else {
                format!(
                    "default: {}, {targets}",
                    cfg_branch_target(*fallback, ssa_live_ins, ssa_locals, body)?
                )
            };
            output.push_str(&format!(
                "{indentation}cf.switch %v{} : {}, [{targets}]\n",
                discriminant.0,
                mlir_type(&value_type(module, *discriminant)?)?,
            ));
        }
        severian_lir::Terminator::Call {
            function: callee,
            arguments,
            destination,
            target,
        } => {
            let callee = function(module, *callee)?;
            if arguments.len() != callee.parameter_types.len() {
                return Err(MlirError::UnsupportedOperation(format!(
                    "call to {} has {} arguments for {} parameters",
                    function_symbol(callee),
                    arguments.len(),
                    callee.parameter_types.len()
                )));
            }
            let mut lowered_arguments = Vec::with_capacity(arguments.len());
            for (index, (value, target_type)) in
                arguments.iter().zip(&callee.parameter_types).enumerate()
            {
                let source_type = value_type(module, *value)?;
                let source = format!("%v{}", value.0);
                let converted = format!("%call_arg_{}_{}", target.0, index);
                let lowered = if source_type == *target_type {
                    source
                } else if matches!(source_type, LoweredType::Tensor { .. })
                    && matches!(target_type, LoweredType::Tensor { .. })
                {
                    output.push_str(&format!(
                        "{indentation}{converted} = tensor.cast {source} : {} to {}\n",
                        mlir_type(&source_type)?,
                        mlir_type(target_type)?
                    ));
                    converted
                } else {
                    render_named_conversion(
                        output,
                        &source,
                        &source_type,
                        target_type,
                        &converted,
                        indent,
                    )?
                };
                lowered_arguments.push(lowered);
            }
            let argument_values = lowered_arguments.join(", ");
            let argument_types = callee
                .parameter_types
                .iter()
                .map(mlir_type)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            if callee.result == LoweredType::Unit {
                output.push_str(&format!(
                    "{indentation}func.call @{}({argument_values}) : ({argument_types}) -> ()\n",
                    function_symbol(callee)
                ));
                output.push_str(&format!(
                    "{indentation}cf.br {}\n",
                    cfg_branch_target(*target, ssa_live_ins, ssa_locals, body)?
                ));
                return Ok(());
            }
            if let Some(destination) = destination {
                let result_type = mlir_type(&callee.result)?;
                output.push_str(&format!(
                    "{indentation}%call_result_{} = func.call @{}({argument_values}) : ({argument_types}) -> {result_type}\n",
                    target.0,
                    function_symbol(callee),
                ));
                if matches!(callee.result, LoweredType::Tensor { .. }) {
                    let severian_lir::PlaceBase::Local(local) = destination.base else {
                        return Err(MlirError::UnsupportedOperation(
                            "tensor call result must target a local place".into(),
                        ));
                    };
                    if !destination.projection.is_empty() {
                        return Err(MlirError::UnsupportedOperation(
                            "tensor call result cannot target a projected place".into(),
                        ));
                    }
                    let local_type = &body.locals[local.0 as usize].ty;
                    let boxed = format!("%call_result_box_{}", target.0);
                    render_tensor_aggregate_box(
                        output,
                        &format!("%call_result_{}", target.0),
                        &callee.result,
                        local_type,
                        &boxed,
                        &format!("call_result_tensor_{}", target.0),
                        indent,
                    )?;
                    output.push_str(&format!(
                        "{indentation}llvm.store {boxed}, %local{} : !llvm.ptr, !llvm.ptr\n",
                        local.0
                    ));
                } else if is_ssa_local_type(&callee.result) {
                    let severian_lir::PlaceBase::Local(local) = destination.base else {
                        return Err(MlirError::UnsupportedOperation(
                            "tensor call result must target a local SSA place".into(),
                        ));
                    };
                    if !destination.projection.is_empty() {
                        return Err(MlirError::UnsupportedOperation(
                            "tensor call result cannot target a projected place".into(),
                        ));
                    }
                    let local_type = &body.locals[local.0 as usize].ty;
                    let result_value = format!("%call_result_{}", target.0);
                    let stored = if callee.result == *local_type {
                        result_value
                    } else if matches!(callee.result, LoweredType::Tensor { .. })
                        && matches!(local_type, LoweredType::Tensor { .. })
                    {
                        let converted = format!("%call_result_cast_{}", target.0);
                        output.push_str(&format!(
                            "{indentation}{converted} = tensor.cast {result_value} : {} to {}\n",
                            mlir_type(&callee.result)?,
                            mlir_type(local_type)?
                        ));
                        converted
                    } else {
                        render_named_conversion(
                            output,
                            &result_value,
                            &callee.result,
                            local_type,
                            &format!("%call_result_cast_{}", target.0),
                            indent,
                        )?
                    };
                    ssa_locals.insert(local, stored);
                } else {
                    output.push_str(&format!(
                        "{indentation}llvm.store %call_result_{}, {} : {result_type}, !llvm.ptr\n",
                        target.0,
                        cfg_place_address(destination)?
                    ));
                }
            } else {
                output.push_str(&format!(
                    "{indentation}func.call @{}({argument_values}) : ({argument_types}) -> ()\n",
                    function_symbol(callee)
                ));
            }
            output.push_str(&format!(
                "{indentation}cf.br {}\n",
                cfg_branch_target(*target, ssa_live_ins, ssa_locals, body)?
            ));
        }
        severian_lir::Terminator::Return(value) => {
            if let Some(value) = value {
                output.push_str(&format!(
                    "{indentation}return %v{} : {}\n",
                    value.0,
                    mlir_type(&value_type(module, *value)?)?
                ));
            } else {
                output.push_str(&format!("{indentation}return\n"));
            }
        }
        severian_lir::Terminator::Throw(value) => {
            let _ = value;
            output.push_str(&format!("{indentation}llvm.unreachable\n"));
        }
        severian_lir::Terminator::Unreachable => {
            output.push_str(&format!("{indentation}llvm.unreachable\n"));
        }
    }
    Ok(())
}

fn cfg_place_address(place: &severian_lir::Place) -> Result<String, MlirError> {
    if !place.projection.is_empty() {
        return Err(MlirError::UnsupportedOperation(format!(
            "projected CFG place {place:?}"
        )));
    }
    Ok(match place.base {
        severian_lir::PlaceBase::Local(local) => format!("%local{}", local.0),
        severian_lir::PlaceBase::Global(global) => format!("%global{}_address", global.0),
    })
}

fn cfg_place_base_address(place: &severian_lir::Place) -> String {
    match place.base {
        severian_lir::PlaceBase::Local(local) => format!("%local{}", local.0),
        severian_lir::PlaceBase::Global(global) => format!("%global{}_address", global.0),
    }
}

fn cfg_place_base_type(
    module: &Module,
    body: &severian_lir::CfgBody,
    place: &severian_lir::Place,
) -> Result<LoweredType, MlirError> {
    match place.base {
        severian_lir::PlaceBase::Local(local) => body
            .locals
            .get(local.0 as usize)
            .map(|local| local.ty.clone())
            .ok_or_else(|| MlirError::UnsupportedOperation(format!("unknown local {}", local.0))),
        severian_lir::PlaceBase::Global(global) => module
            .storage_globals
            .get(global.0 as usize)
            .map(|global| global.ty.clone())
            .ok_or_else(|| MlirError::UnsupportedOperation(format!("unknown global {}", global.0))),
    }
}

fn render_cfg_unary(
    output: &mut String,
    module: &Module,
    operator: UnaryOperation,
    operand: ValueId,
    result: ValueId,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let ty = value_type(module, operand)?;
    match operator {
        UnaryOperation::Positive => output.push_str(&format!(
            "{indentation}%v{} = builtin.unrealized_conversion_cast %v{} : {} to {}\n",
            result.0,
            operand.0,
            mlir_type(&ty)?,
            mlir_type(&ty)?
        )),
        UnaryOperation::Negative => {
            let zero = format!("v{}_zero", result.0);
            let zero_literal = if matches!(ty, LoweredType::Float { .. }) {
                "0.0"
            } else {
                "0"
            };
            output.push_str(&format!(
                "{indentation}%{zero} = arith.constant {zero_literal} : {}\n",
                mlir_type(&ty)?
            ));
            output.push_str(&format!(
                "{indentation}%v{} = {} %{zero}, %v{} : {}\n",
                result.0,
                if matches!(ty, LoweredType::Float { .. }) {
                    "arith.subf"
                } else {
                    "arith.subi"
                },
                operand.0,
                mlir_type(&ty)?
            ));
        }
        UnaryOperation::Not => {
            output.push_str(&format!(
                "{indentation}%v{}_not = arith.constant true\n",
                result.0
            ));
            output.push_str(&format!(
                "{indentation}%v{} = arith.xori %v{}, %v{}_not : i1\n",
                result.0, operand.0, result.0
            ));
        }
    }
    Ok(())
}

fn render_conversion(
    output: &mut String,
    module: &Module,
    operand: ValueId,
    result: ValueId,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let source = value_type(module, operand)?;
    let target = value_type(module, result)?;
    let source_type = mlir_type(&source)?;
    let target_type = mlir_type(&target)?;

    if source_type == target_type {
        output.push_str(&format!(
            "{indentation}%v{} = builtin.unrealized_conversion_cast %v{} : {source_type} to {target_type}\n",
            result.0, operand.0
        ));
        return Ok(());
    }

    let instruction = match (&source, &target) {
        (
            LoweredType::Integer {
                bits: source_bits,
                signed,
            },
            LoweredType::Integer {
                bits: target_bits, ..
            },
        ) if target_bits > source_bits => {
            if *signed {
                "arith.extsi"
            } else {
                "arith.extui"
            }
        }
        (
            LoweredType::Integer {
                bits: source_bits, ..
            },
            LoweredType::Integer {
                bits: target_bits, ..
            },
        ) if target_bits < source_bits => "arith.trunci",
        (LoweredType::Integer { signed: true, .. }, LoweredType::Float { .. }) => "arith.sitofp",
        (LoweredType::Integer { signed: false, .. }, LoweredType::Float { .. }) => "arith.uitofp",
        (LoweredType::Float { .. }, LoweredType::Integer { signed: true, .. }) => "arith.fptosi",
        (LoweredType::Float { .. }, LoweredType::Integer { signed: false, .. }) => "arith.fptoui",
        (
            LoweredType::Float {
                format: source_format,
            },
            LoweredType::Float {
                format: target_format,
            },
        ) => {
            let source_bits = float_bits(*source_format);
            let target_bits = float_bits(*target_format);
            if target_bits > source_bits {
                "arith.extf"
            } else if target_bits < source_bits {
                "arith.truncf"
            } else {
                output.push_str(&format!(
                    "{indentation}%v{}_wide = arith.extf %v{} : {source_type} to f32\n",
                    result.0, operand.0
                ));
                output.push_str(&format!(
                    "{indentation}%v{} = arith.truncf %v{}_wide : f32 to {target_type}\n",
                    result.0, result.0
                ));
                return Ok(());
            }
        }
        _ => {
            return Err(MlirError::UnsupportedOperation(format!(
                "numeric conversion from {source:?} to {target:?}"
            )))
        }
    };
    output.push_str(&format!(
        "{indentation}%v{} = {instruction} %v{} : {source_type} to {target_type}\n",
        result.0, operand.0
    ));
    Ok(())
}

fn render_named_conversion(
    output: &mut String,
    source_value: &str,
    source: &LoweredType,
    target: &LoweredType,
    result_value: &str,
    indent: usize,
) -> Result<String, MlirError> {
    if source == target {
        return Ok(source_value.to_owned());
    }
    let instruction = match (source, target) {
        (
            LoweredType::Integer {
                bits: source_bits,
                signed,
            },
            LoweredType::Integer {
                bits: target_bits, ..
            },
        ) if target_bits > source_bits => {
            if *signed {
                "arith.extsi"
            } else {
                "arith.extui"
            }
        }
        (
            LoweredType::Integer {
                bits: source_bits, ..
            },
            LoweredType::Integer {
                bits: target_bits, ..
            },
        ) if target_bits < source_bits => "arith.trunci",
        (LoweredType::Integer { signed: true, .. }, LoweredType::Float { .. }) => "arith.sitofp",
        (LoweredType::Integer { signed: false, .. }, LoweredType::Float { .. }) => "arith.uitofp",
        (LoweredType::Float { .. }, LoweredType::Integer { signed: true, .. }) => "arith.fptosi",
        (LoweredType::Float { .. }, LoweredType::Integer { signed: false, .. }) => "arith.fptoui",
        (LoweredType::Float { format: source }, LoweredType::Float { format: target })
            if float_bits(*target) > float_bits(*source) =>
        {
            "arith.extf"
        }
        (LoweredType::Float { format: source }, LoweredType::Float { format: target })
            if float_bits(*target) < float_bits(*source) =>
        {
            "arith.truncf"
        }
        _ => {
            return Err(MlirError::UnsupportedOperation(format!(
                "aggregate conversion from {source:?} to {target:?}"
            )))
        }
    };
    output.push_str(&format!(
        "{}{result_value} = {instruction} {source_value} : {} to {}\n",
        " ".repeat(indent),
        mlir_type(source)?,
        mlir_type(target)?
    ));
    Ok(result_value.to_owned())
}

fn float_bits(format: LoweredFloatFormat) -> u16 {
    match format {
        LoweredFloatFormat::Float8E4M3Fn | LoweredFloatFormat::Float8E5M2 => 8,
        LoweredFloatFormat::Ieee(bits) => bits,
        LoweredFloatFormat::BrainFloat16 => 16,
    }
}

fn render_runtime_call(
    output: &mut String,
    module: &Module,
    symbol: &str,
    arguments: &[ValueId],
    result: Option<ValueId>,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let aggregate_abi = symbol.contains("_aggregate");
    let tag = result
        .or_else(|| arguments.first().copied())
        .map_or(0, |value| value.0);
    let mut argument_values = Vec::with_capacity(arguments.len());
    let mut argument_types = Vec::with_capacity(arguments.len());
    for (index, value) in arguments.iter().copied().enumerate() {
        let ty = value_type(module, value)?;
        if matches!(ty, LoweredType::Tensor { .. }) {
            return Err(MlirError::UnsupportedOperation(format!(
                "native runtime symbol `{symbol}` cannot receive an MLIR builtin tensor; use a !llvm.ptr StorageViewAbi at the host boundary"
            )));
        }
        if aggregate_abi && matches!(ty, LoweredType::Aggregate(_)) {
            let spelling = mlir_type(&ty)?;
            let (size, _) = lowered_type_layout(module, &ty, &mut BTreeSet::new())?;
            output.push_str(&format!(
                "{indentation}%runtime_box_one_{tag}_{index} = arith.constant 1 : i64\n"
            ));
            output.push_str(&format!(
                "{indentation}%runtime_box_slot_{tag}_{index} = llvm.alloca %runtime_box_one_{tag}_{index} x {spelling} : (i64) -> !llvm.ptr\n"
            ));
            output.push_str(&format!(
                "{indentation}llvm.store %v{}, %runtime_box_slot_{tag}_{index} : {spelling}, !llvm.ptr\n",
                value.0
            ));
            output.push_str(&format!(
                "{indentation}%runtime_box_size_{tag}_{index} = arith.constant {size} : i64\n"
            ));
            output.push_str(&format!(
                "{indentation}%runtime_box_{tag}_{index} = func.call @__sev_aggregate_box(%runtime_box_slot_{tag}_{index}, %runtime_box_size_{tag}_{index}) : (!llvm.ptr, i64) -> !llvm.ptr\n"
            ));
            argument_values.push(format!("%runtime_box_{tag}_{index}"));
            argument_types.push("!llvm.ptr".into());
        } else {
            argument_values.push(format!("%v{}", value.0));
            argument_types.push(mlir_type(&ty)?);
        }
    }
    let arguments_text = argument_values.join(", ");
    let argument_types = argument_types.join(", ");
    if let Some(result) = result {
        let result_ty = value_type(module, result)?;
        if matches!(result_ty, LoweredType::Tensor { .. }) {
            return Err(MlirError::UnsupportedOperation(format!(
                "native runtime symbol `{symbol}` cannot return an MLIR builtin tensor; declare the C void pointer result as !llvm.ptr StorageViewAbi and specialize it before compute lowering"
            )));
        }
        if aggregate_abi && matches!(result_ty, LoweredType::Aggregate(_)) {
            let spelling = mlir_type(&result_ty)?;
            output.push_str(&format!(
                "{indentation}%runtime_box_result_{} = func.call @{symbol}({arguments_text}) : ({argument_types}) -> !llvm.ptr\n",
                result.0
            ));
            output.push_str(&format!(
                "{indentation}%v{} = llvm.load %runtime_box_result_{} : !llvm.ptr -> {spelling}\n",
                result.0, result.0
            ));
            return Ok(());
        }
        output.push_str(&format!(
            "{indentation}%v{} = func.call @{symbol}({arguments_text}) : ({argument_types}) -> {}\n",
            result.0,
            mlir_type(&result_ty)?
        ));
    } else {
        output.push_str(&format!(
            "{indentation}func.call @{symbol}({arguments_text}) : ({argument_types}) -> ()\n"
        ));
    }
    Ok(())
}

fn runtime_abi_type(ty: LoweredType, aggregate_abi: bool) -> LoweredType {
    if aggregate_abi && matches!(ty, LoweredType::Aggregate(_)) {
        LoweredType::String
    } else {
        ty
    }
}

fn lowered_type_layout(
    module: &Module,
    ty: &LoweredType,
    visiting: &mut BTreeSet<u32>,
) -> Result<(u64, u64), MlirError> {
    let scalar = match ty {
        LoweredType::Integer { bits, .. } => Some(u64::from(*bits).div_ceil(8).max(1)),
        LoweredType::Float { format } => Some(u64::from(float_bits(*format)).div_ceil(8).max(1)),
        LoweredType::Boolean | LoweredType::None | LoweredType::Unit => Some(1),
        LoweredType::String | LoweredType::Bytes => Some(8),
        LoweredType::Arguments => return Ok((16, 8)),
        LoweredType::Task(_) => return Ok((8, 8)),
        LoweredType::Aggregate(_) => None,
        LoweredType::Tensor { .. } => {
            return Err(MlirError::UnsupportedOperation(
                "tensor values do not use the aggregate runtime ABI".into(),
            ))
        }
    };
    if let Some(size) = scalar {
        return Ok((size, size.clamp(1, 8)));
    }
    let LoweredType::Aggregate(id) = ty else {
        unreachable!("non-scalar layout is aggregate")
    };
    if !visiting.insert(*id) {
        return Err(MlirError::UnsupportedOperation(format!(
            "aggregate class {id} has a recursive inline layout"
        )));
    }
    let declaration = module
        .classes
        .iter()
        .find(|declaration| declaration.id == *id)
        .ok_or_else(|| MlirError::UnsupportedOperation(format!("unknown aggregate class {id}")))?;
    let mut size = 0u64;
    let mut aggregate_alignment = 1u64;
    for field in &declaration.fields {
        let (field_size, alignment) = lowered_type_layout(module, &field.ty, visiting)?;
        aggregate_alignment = aggregate_alignment.max(alignment);
        size = size.div_ceil(alignment) * alignment;
        size = size.saturating_add(field_size);
    }
    visiting.remove(&id);
    Ok((
        size.div_ceil(aggregate_alignment) * aggregate_alignment,
        aggregate_alignment,
    ))
}

fn render_assert(
    output: &mut String,
    module: &Module,
    condition: ValueId,
    _message: Option<ValueId>,
    _location: Option<&severian_lir::AssertionLocation>,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let _ = value_type(module, condition)?;
    output.push_str(&format!(
        "{indentation}%assert_message_{} = llvm.mlir.addressof @__sev_assert_message_{} : !llvm.ptr\n",
        condition.0, condition.0
    ));
    output.push_str(&format!(
        "{indentation}func.call @__sev_assert(%v{}, %assert_message_{}) : (i1, !llvm.ptr) -> ()\n",
        condition.0, condition.0
    ));
    Ok(())
}

fn binary_mnemonic(operator: BinaryOperation, ty: &LoweredType) -> Result<String, MlirError> {
    let floating = matches!(ty, LoweredType::Float { .. });
    let signed = matches!(ty, LoweredType::Integer { signed: true, .. });
    Ok(match operator {
        BinaryOperation::BitwiseOr | BinaryOperation::Or => "arith.ori".into(),
        BinaryOperation::BitwiseAnd | BinaryOperation::And => "arith.andi".into(),
        BinaryOperation::BitwiseXor => "arith.xori".into(),
        BinaryOperation::Add => if floating { "arith.addf" } else { "arith.addi" }.into(),
        BinaryOperation::Subtract => if floating { "arith.subf" } else { "arith.subi" }.into(),
        BinaryOperation::Multiply => if floating { "arith.mulf" } else { "arith.muli" }.into(),
        BinaryOperation::Divide => if floating {
            "arith.divf"
        } else if signed {
            "arith.divsi"
        } else {
            "arith.divui"
        }
        .into(),
        BinaryOperation::FloorDivide => if signed {
            "arith.floordivsi"
        } else {
            "arith.divui"
        }
        .into(),
        BinaryOperation::Remainder => if floating {
            "arith.remf"
        } else if signed {
            "arith.remsi"
        } else {
            "arith.remui"
        }
        .into(),
        BinaryOperation::Power => "math.ipowi".into(),
        BinaryOperation::ShiftLeft => "arith.shli".into(),
        BinaryOperation::ShiftRight => if signed {
            "arith.shrsi"
        } else {
            "arith.shrui"
        }
        .into(),
        BinaryOperation::Equal => if floating {
            "arith.cmpf oeq,"
        } else {
            "arith.cmpi eq,"
        }
        .into(),
        BinaryOperation::NotEqual => if floating {
            "arith.cmpf one,"
        } else {
            "arith.cmpi ne,"
        }
        .into(),
        BinaryOperation::Less => if floating {
            "arith.cmpf olt,"
        } else if signed {
            "arith.cmpi slt,"
        } else {
            "arith.cmpi ult,"
        }
        .into(),
        BinaryOperation::LessEqual => if floating {
            "arith.cmpf ole,"
        } else if signed {
            "arith.cmpi sle,"
        } else {
            "arith.cmpi ule,"
        }
        .into(),
        BinaryOperation::Greater => if floating {
            "arith.cmpf ogt,"
        } else if signed {
            "arith.cmpi sgt,"
        } else {
            "arith.cmpi ugt,"
        }
        .into(),
        BinaryOperation::GreaterEqual => if floating {
            "arith.cmpf oge,"
        } else if signed {
            "arith.cmpi sge,"
        } else {
            "arith.cmpi uge,"
        }
        .into(),
        unsupported => {
            return Err(MlirError::UnsupportedOperation(format!(
                "binary operation {unsupported:?}"
            )));
        }
    })
}

fn mlir_trait_type(ty: &severian_lir::TraitType) -> Result<String, MlirError> {
    match ty {
        severian_lir::TraitType::SelfType => Ok("Self".into()),
        severian_lir::TraitType::Concrete(ty) => mlir_type(ty),
        severian_lir::TraitType::Symbolic(name) => Ok(name.clone()),
    }
}

fn render_block(
    output: &mut String,
    module: &Module,
    block: &Block,
    indent: usize,
    function_result: Option<&LoweredType>,
    coverage_ordinal: &mut usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    for operation in &block.operations {
        match operation {
            Operation::Coverage { key } => {
                let symbol = coverage_symbol(key);
                let value = format!("{symbol}_{}", *coverage_ordinal);
                *coverage_ordinal += 1;
                output.push_str(&format!(
                    "{indentation}%{value} = llvm.mlir.addressof @{symbol} : !llvm.ptr\n"
                ));
                output.push_str(&format!(
                    "{indentation}func.call @__sev_coverage_hit(%{value}) : (!llvm.ptr) -> ()\n"
                ));
            }
            Operation::Constant { value, result } => {
                let ty = value_type(module, *result)?;
                let spelling = mlir_type(&ty)?;
                if matches!(value, Constant::None) && spelling == "!llvm.ptr" {
                    output.push_str(&format!(
                        "{indentation}%v{} = llvm.mlir.zero : !llvm.ptr\n",
                        result.0
                    ));
                    continue;
                }
                let literal = match value {
                    Constant::Integer(value) => value.clone(),
                    Constant::Float(value) => mlir_float_literal(value),
                    Constant::Boolean(true) => "1".into(),
                    Constant::Boolean(false) => "0".into(),
                    Constant::String(_) => {
                        output.push_str(&format!(
                            "{indentation}%v{} = llvm.mlir.addressof @{} : !llvm.ptr\n",
                            result.0,
                            string_symbol(*result),
                        ));
                        continue;
                    }
                    other => {
                        return Err(MlirError::UnsupportedOperation(format!(
                            "MLIR constant lowering is unavailable for {other:?}"
                        )))
                    }
                };
                output.push_str(&format!(
                    "{indentation}%v{} = arith.constant {literal} : {spelling}\n",
                    result.0
                ));
            }
            Operation::Unary {
                operator,
                operand,
                result,
            } => {
                let lowered_type = value_type(module, *result)?;
                let ty = mlir_type(&lowered_type)?;
                match (operator, lowered_type) {
                    (UnaryOperation::Not, LoweredType::Boolean) => {
                        output.push_str(&format!(
                            "{indentation}%v{}_not = arith.constant true\n",
                            result.0
                        ));
                        output.push_str(&format!(
                            "{indentation}%v{} = arith.xori %v{}, %v{}_not : i1\n",
                            result.0, operand.0, result.0
                        ));
                    }
                    _ => {
                        return Err(MlirError::UnsupportedOperation(format!(
                            "MLIR unary {operator:?} for %v{} -> %v{} : {ty} requires a dedicated lowering",
                            operand.0, result.0
                        )));
                    }
                }
            }
            Operation::Convert {
                operand, result, ..
            } => {
                render_conversion(output, module, *operand, *result, indent)?;
            }
            Operation::Binary {
                operator,
                left,
                right,
                result,
            } => {
                let input_type = value_type(module, *left)?;
                let spelling = mlir_type(&input_type)?;
                let instruction = mlir_binary(*operator, &input_type)?;
                output.push_str(&format!(
                    "{indentation}%v{} = {instruction} %v{}, %v{} : {spelling}\n",
                    result.0, left.0, right.0
                ));
            }
            Operation::Aggregate {
                class,
                fields,
                result,
            } => {
                let ty = mlir_type(&LoweredType::Aggregate(*class))?;
                if fields.is_empty() {
                    output.push_str(&format!(
                        "{indentation}%v{} = llvm.mlir.undef : {ty}\n",
                        result.0
                    ));
                    continue;
                }
                output.push_str(&format!(
                    "{indentation}%v{}_aggregate_0 = llvm.mlir.undef : {ty}\n",
                    result.0
                ));
                for (index, field) in fields.iter().enumerate() {
                    let input = if index == 0 {
                        format!("%v{}_aggregate_0", result.0)
                    } else {
                        format!("%v{}_aggregate_{index}", result.0)
                    };
                    let result_name = if index + 1 == fields.len() {
                        format!("%v{}", result.0)
                    } else {
                        format!("%v{}_aggregate_{}", result.0, index + 1)
                    };
                    output.push_str(&format!(
                        "{indentation}{result_name} = llvm.insertvalue %v{}, {input}[{index}] : {ty}\n",
                        field.0
                    ));
                }
            }
            Operation::FieldGet {
                object,
                field,
                result,
            } => {
                let ty = mlir_type(&value_type(module, *object)?)?;
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.extractvalue %v{}[{field}] : {ty}\n",
                    result.0, object.0
                ));
            }
            Operation::FieldSet {
                object,
                field,
                value,
                result,
            } => {
                let ty = mlir_type(&value_type(module, *object)?)?;
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.insertvalue %v{}, %v{}[{field}] : {ty}\n",
                    result.0, value.0, object.0
                ));
            }
            Operation::Load { place, result } => {
                let ty = value_type(module, *result)?;
                output.push_str(&format!(
                    "{indentation}%v{} = llvm.load {} : {}\n",
                    result.0,
                    cfg_place_address(place)?,
                    mlir_type(&ty)?
                ));
            }
            Operation::AddressOf { place, result } => {
                output.push_str(&format!(
                    "{indentation}%v{} = builtin.unrealized_conversion_cast {} : !llvm.ptr to !llvm.ptr\n",
                    result.0,
                    cfg_place_address(place)?
                ));
            }
            Operation::Store { place, value } => {
                let ty = value_type(module, *value)?;
                output.push_str(&format!(
                    "{indentation}llvm.store %v{}, {} : {}\n",
                    value.0,
                    cfg_place_address(place)?,
                    mlir_type(&ty)?
                ));
            }
            Operation::Call {
                function: target,
                arguments,
                result,
            } => {
                let target = function(module, *target)?;
                if arguments.len() != target.parameter_types.len() {
                    return Err(MlirError::UnsupportedOperation(format!(
                        "call to {} has {} arguments for {} parameters",
                        function_symbol(target),
                        arguments.len(),
                        target.parameter_types.len()
                    )));
                }
                let mut lowered_arguments = Vec::with_capacity(arguments.len());
                for (index, (value, target_type)) in
                    arguments.iter().zip(&target.parameter_types).enumerate()
                {
                    let source_type = value_type(module, *value)?;
                    let source = format!("%v{}", value.0);
                    let converted = format!("%call_arg_{}_{}", result.0, index);
                    let lowered = if source_type == *target_type {
                        source
                    } else if matches!(source_type, LoweredType::Tensor { .. })
                        && matches!(target_type, LoweredType::Tensor { .. })
                    {
                        output.push_str(&format!(
                            "{indentation}{converted} = tensor.cast {source} : {} to {}\n",
                            mlir_type(&source_type)?,
                            mlir_type(target_type)?
                        ));
                        converted
                    } else {
                        render_named_conversion(
                            output,
                            &source,
                            &source_type,
                            target_type,
                            &converted,
                            indent,
                        )?
                    };
                    lowered_arguments.push(lowered);
                }
                let arguments = lowered_arguments.join(", ");
                let argument_types = argument_types(module, target)?;
                if target.result == LoweredType::Unit {
                    output.push_str(&format!(
                        "{indentation}func.call @{}({arguments}) : ({argument_types}) -> ()\n",
                        function_symbol(target),
                    ));
                } else {
                    let result_type = mlir_type(&value_type(module, *result)?)?;
                    output.push_str(&format!(
                        "{indentation}%v{} = func.call @{}({arguments}) : ({argument_types}) -> {result_type}\n",
                        result.0,
                        function_symbol(target),
                    ));
                }
            }
            Operation::Spawn {
                function: target,
                arguments,
                result,
                owner,
                locked,
            } => {
                let target = function(module, *target)?;
                let arguments_text = arguments
                    .iter()
                    .map(|value| format!("%v{}", value.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let argument_types = argument_types(module, target)?;
                let owner = match owner {
                    severian_lir::TaskOwner::SelfScope => "self",
                    severian_lir::TaskOwner::Runtime => "runtime",
                    severian_lir::TaskOwner::Inferred => "inferred",
                };
                output.push_str(&format!(
                    "{indentation}// severian.task owner={owner} locked={locked}\n"
                ));
                let attributes = format!(
                    "attributes {{severian.owner = \"{owner}\", severian.locked = {locked}}}"
                );
                if target.result == LoweredType::Unit {
                    output.push_str(&format!(
                        "{indentation}%v{} = async.execute {attributes} {{\n",
                        result.0,
                    ));
                    if *locked {
                        output.push_str(&format!(
                            "{}func.call @__sev_task_lock() : () -> ()\n",
                            " ".repeat(indent + 2)
                        ));
                    }
                    output.push_str(&format!(
                        "{}func.call @{}({arguments_text}) : ({argument_types}) -> ()\n",
                        " ".repeat(indent + 2),
                        function_symbol(target),
                    ));
                    if *locked {
                        output.push_str(&format!(
                            "{}func.call @__sev_task_unlock() : () -> ()\n",
                            " ".repeat(indent + 2)
                        ));
                    }
                    output.push_str(&format!("{}async.yield\n", " ".repeat(indent + 2)));
                } else {
                    let result_type = mlir_type(&value_type(module, *result)?)?;
                    output.push_str(&format!(
                        "{indentation}%task_token{}, %v{} = async.execute -> !async.value<{result_type}> {attributes} {{\n",
                        result.0, result.0,
                    ));
                    if *locked {
                        output.push_str(&format!(
                            "{}func.call @__sev_task_lock() : () -> ()\n",
                            " ".repeat(indent + 2)
                        ));
                    }
                    output.push_str(&format!(
                        "{}%task_value{} = func.call @{}({arguments_text}) : ({argument_types}) -> {result_type}\n",
                        " ".repeat(indent + 2),
                        result.0,
                        function_symbol(target),
                    ));
                    if *locked {
                        output.push_str(&format!(
                            "{}func.call @__sev_task_unlock() : () -> ()\n",
                            " ".repeat(indent + 2)
                        ));
                    }
                    output.push_str(&format!(
                        "{}async.yield %task_value{} : {result_type}\n",
                        " ".repeat(indent + 2),
                        result.0
                    ));
                }
                output.push_str(&format!("{indentation}}}\n"));
            }
            Operation::SpawnFieldUpdate { .. } => {
                return Err(MlirError::UnsupportedOperation(
                    "async field updates require CFG lowering".into(),
                ));
            }
            Operation::Await { task, result } => {
                let ty = value_type(module, *result)?;
                if ty == LoweredType::Unit {
                    output.push_str(&format!(
                        "{indentation}async.await %v{} : !async.token\n",
                        task.0
                    ));
                    output.push_str(&format!(
                        "{indentation}%v{} = arith.constant 0 : i8\n",
                        result.0
                    ));
                } else {
                    let spelling = mlir_type(&ty)?;
                    output.push_str(&format!(
                        "{indentation}%v{} = async.await %v{} : !async.value<{spelling}>\n",
                        result.0, task.0
                    ));
                }
            }
            Operation::RuntimeCall {
                symbol,
                arguments,
                result,
            } => render_runtime_call(output, module, symbol, arguments, *result, indent)?,
            Operation::Return { value } => {
                let expected = function_result.ok_or_else(|| {
                    MlirError::UnsupportedOperation(
                        "return is not valid in module initialization".into(),
                    )
                })?;
                match (value, expected) {
                    (None, LoweredType::Unit) => {
                        output.push_str(&format!("{indentation}return\n"));
                    }
                    (Some(value), expected) if *expected != LoweredType::Unit => {
                        let actual = value_type(module, *value)?;
                        if actual != *expected {
                            return Err(MlirError::SignatureMismatch);
                        }
                        output.push_str(&format!(
                            "{indentation}return %v{} : {}\n",
                            value.0,
                            mlir_type(&actual)?
                        ));
                    }
                    _ => return Err(MlirError::SignatureMismatch),
                }
            }
            Operation::Assert {
                condition,
                message,
                location,
            } => {
                let custom = message.and_then(|message| constant_string(module, message));
                let failure = location.as_ref().map_or_else(
                    || custom.unwrap_or("assertion failed").to_owned(),
                    |location| {
                        let mut failure = format!(
                            "{}:{}:{}: assertion failed: {}",
                            location.file, location.line, location.column, location.expression
                        );
                        if let Some(custom) = custom {
                            failure.push_str(": ");
                            failure.push_str(custom);
                        }
                        failure
                    },
                );
                output.push_str(&format!(
                    "{indentation}cf.assert %v{}, \"{}\"\n",
                    condition.0,
                    mlir_string(&failure)
                ));
            }
            Operation::If {
                condition,
                then_block,
                else_block,
            } => {
                if block_contains_return(then_block) || block_contains_return(else_block) {
                    return Err(MlirError::UnsupportedOperation(
                        "return inside if requires CFG lowering".into(),
                    ));
                }
                output.push_str(&format!("{indentation}scf.if %v{} {{\n", condition.0));
                render_block(
                    output,
                    module,
                    then_block,
                    indent + 2,
                    function_result,
                    coverage_ordinal,
                )?;
                output.push_str(&format!("{}scf.yield\n", " ".repeat(indent + 2)));
                if !else_block.operations.is_empty() {
                    output.push_str(&format!("{indentation}}} else {{\n"));
                    render_block(
                        output,
                        module,
                        else_block,
                        indent + 2,
                        function_result,
                        coverage_ordinal,
                    )?;
                    output.push_str(&format!("{}scf.yield\n", " ".repeat(indent + 2)));
                }
                output.push_str(&format!("{indentation}}}\n"));
            }
            Operation::While {
                condition_block,
                condition,
                body,
            } => {
                if block_contains_return(body) {
                    return Err(MlirError::UnsupportedOperation(
                        "return inside while requires CFG lowering".into(),
                    ));
                }
                output.push_str(&format!("{indentation}scf.while : () -> () {{\n"));
                render_block(
                    output,
                    module,
                    condition_block,
                    indent + 2,
                    function_result,
                    coverage_ordinal,
                )?;
                output.push_str(&format!(
                    "{}scf.condition(%v{})\n",
                    " ".repeat(indent + 2),
                    condition.0
                ));
                output.push_str(&format!("{indentation}}} do {{\n"));
                render_block(
                    output,
                    module,
                    body,
                    indent + 2,
                    function_result,
                    coverage_ordinal,
                )?;
                output.push_str(&format!("{}scf.yield\n", " ".repeat(indent + 2)));
                output.push_str(&format!("{indentation}}}\n"));
            }
            Operation::Break | Operation::Continue => {
                return Err(MlirError::UnsupportedOperation(
                    "loop control requires CFG-to-MLIR branch lowering".into(),
                ));
            }
            Operation::ArtifactCall {
                artifact,
                inputs,
                outputs,
            } => {
                let symbol = artifact_symbol(*artifact);
                let arguments = inputs
                    .iter()
                    .map(|value| format!("%v{}", value.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let input_types = inputs
                    .iter()
                    .map(|value| mlir_type(&value_type(module, *value)?))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                match outputs.as_slice() {
                    [] => output.push_str(&format!(
                        "{indentation}func.call @{symbol}({arguments}) : ({input_types}) -> ()\n"
                    )),
                    [result] => {
                        let ty = mlir_type(&value_type(module, *result)?)?;
                        output.push_str(&format!(
                            "{indentation}%v{} = func.call @{symbol}({arguments}) : ({input_types}) -> {ty}\n",
                            result.0
                        ));
                    }
                    results => {
                        let result_names = results
                            .iter()
                            .map(|result| format!("%v{}", result.0))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let result_types = results
                            .iter()
                            .map(|result| mlir_type(&value_type(module, *result)?))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(", ");
                        output.push_str(&format!(
                            "{indentation}{result_names} = func.call @{symbol}({arguments}) : ({input_types}) -> ({result_types})\n"
                        ));
                    }
                }
            }
        }
        if matches!(operation, Operation::Return { .. }) {
            break;
        }
    }
    Ok(())
}

fn argument_types(module: &Module, function: &Function) -> Result<String, MlirError> {
    function
        .parameters
        .iter()
        .map(|parameter| mlir_type(&value_type(module, *parameter)?))
        .collect::<Result<Vec<_>, _>>()
        .map(|types| types.join(", "))
}

fn render_function_declaration(
    output: &mut String,
    module: &Module,
    function: &Function,
) -> Result<(), MlirError> {
    let parameters = function
        .parameters
        .iter()
        .map(|parameter| mlir_type(&value_type(module, *parameter)?))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let result = function_result(&function.result)?;
    output.push_str(&format!(
        "  func.func private @{}({parameters}){result}\n",
        function_symbol(function),
    ));
    Ok(())
}

fn render_function_definition(
    output: &mut String,
    module: &Module,
    function: &Function,
) -> Result<(), MlirError> {
    let parameters = function
        .parameters
        .iter()
        .map(|parameter| {
            Ok(format!(
                "%v{}: {}",
                parameter.0,
                mlir_type(&value_type(module, *parameter)?)?
            ))
        })
        .collect::<Result<Vec<_>, MlirError>>()?
        .join(", ");
    let result = function_result(&function.result)?;
    output.push_str(&format!(
        "  func.func private @{}({parameters}){result} {{\n",
        function_symbol(function),
    ));
    let body = function.body.as_ref().expect("filtered source body");
    let mut coverage_ordinal = 0;
    render_block(
        output,
        module,
        body,
        4,
        Some(&function.result),
        &mut coverage_ordinal,
    )?;
    if !block_terminates(body) {
        if function.result != LoweredType::Unit {
            return Err(MlirError::UnsupportedOperation(format!(
                "function `{}` falls through without returning a value",
                function.name
            )));
        }
        output.push_str("    return\n");
    }
    output.push_str("  }\n");
    Ok(())
}

fn function_result(result: &LoweredType) -> Result<String, MlirError> {
    if *result == LoweredType::Unit {
        Ok(String::new())
    } else {
        Ok(format!(" -> {}", mlir_type(result)?))
    }
}

fn function(module: &Module, id: FunctionId) -> Result<&Function, MlirError> {
    module
        .functions
        .iter()
        .find(|function| function.id == id)
        .ok_or_else(|| MlirError::UnsupportedOperation(format!("unknown LIR function {}", id.0)))
}

fn function_symbol(function: &Function) -> String {
    match &function.linkage {
        FunctionLinkage::Internal => format!("__sev_fn_{}", function.id.0),
        FunctionLinkage::External { symbol } => symbol.clone(),
    }
}

fn all_operations(module: &Module) -> Vec<&Operation> {
    let mut operations = Vec::new();
    collect_operations(&module.initializer, &mut operations);
    for body in module
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref())
    {
        collect_operations(body, &mut operations);
    }
    operations
}

fn collect_operations<'a>(block: &'a Block, operations: &mut Vec<&'a Operation>) {
    for operation in &block.operations {
        operations.push(operation);
        if let Operation::If {
            then_block,
            else_block,
            ..
        } = operation
        {
            collect_operations(then_block, operations);
            collect_operations(else_block, operations);
        } else if let Operation::While {
            condition_block,
            body,
            ..
        } = operation
        {
            collect_operations(condition_block, operations);
            collect_operations(body, operations);
        }
    }
}

fn block_contains_return(block: &Block) -> bool {
    block.operations.iter().any(|operation| match operation {
        Operation::Return { .. } => true,
        Operation::If {
            then_block,
            else_block,
            ..
        } => block_contains_return(then_block) || block_contains_return(else_block),
        Operation::While {
            condition_block,
            body,
            ..
        } => block_contains_return(condition_block) || block_contains_return(body),
        _ => false,
    })
}

fn block_terminates(block: &Block) -> bool {
    block.operations.iter().any(|operation| match operation {
        Operation::Return { .. } => true,
        Operation::If {
            then_block,
            else_block,
            ..
        } => {
            !else_block.operations.is_empty()
                && block_terminates(then_block)
                && block_terminates(else_block)
        }
        _ => false,
    })
}

fn string_symbol(result: ValueId) -> String {
    format!("__sev_string_{}", result.0)
}

fn coverage_symbol(key: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in key.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("__sev_coverage_{hash:016x}")
}

fn mlir_string(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        match byte {
            b'"' => output.push_str("\\22"),
            b'\\' => output.push_str("\\5C"),
            0x20..=0x7e => output.push(char::from(byte)),
            byte => output.push_str(&format!("\\{byte:02X}")),
        }
    }
    output
}

pub(crate) fn artifact_symbol(artifact: ArtifactId) -> String {
    format!("__sev_artifact_{}", artifact.index())
}

fn value_type(module: &Module, id: ValueId) -> Result<LoweredType, MlirError> {
    module
        .values
        .get(id.0 as usize)
        .filter(|value| value.id == id)
        .map(|value| value.ty.clone())
        .ok_or(MlirError::InvalidValue(id))
}

fn constant_string(module: &Module, id: ValueId) -> Option<&str> {
    all_operations(module)
        .into_iter()
        .find_map(|operation| match operation {
            Operation::Constant {
                value: Constant::String(value),
                result,
            } if *result == id => Some(value.as_str()),
            _ => None,
        })
}

fn mlir_float_literal(value: &str) -> String {
    if let Some(fraction) = value.strip_prefix('.') {
        format!("0.{fraction}")
    } else if let Some(exponent) = value.find(['e', 'E']) {
        let (mantissa, exponent) = value.split_at(exponent);
        if mantissa.contains('.') {
            value.to_owned()
        } else {
            format!("{mantissa}.0{exponent}")
        }
    } else {
        value.to_owned()
    }
}

pub(crate) fn mlir_type(ty: &LoweredType) -> Result<String, MlirError> {
    Ok(match ty {
        LoweredType::Integer { bits, .. } => format!("i{bits}"),
        LoweredType::Float {
            format: LoweredFloatFormat::Float8E4M3Fn,
        }
        | LoweredType::Float {
            format: LoweredFloatFormat::Float8E5M2,
        } => "i8".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(16),
        } => "f16".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        } => "f32".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(64),
        } => "f64".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(80),
        } => "f80".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(128),
        } => "f128".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::BrainFloat16,
        } => "bf16".into(),
        unsupported @ LoweredType::Float { .. } => {
            return Err(MlirError::UnsupportedType(unsupported.clone()))
        }
        LoweredType::Boolean => "i1".into(),
        LoweredType::String | LoweredType::Bytes => "!llvm.ptr".into(),
        LoweredType::None | LoweredType::Unit => "i8".into(),
        LoweredType::Arguments => "!llvm.struct<(i32, !llvm.ptr)>".into(),
        LoweredType::Aggregate(id) => format!("!sev_class_{id}"),
        LoweredType::Tensor { element, shape } => {
            let element = mlir_tensor_element(*element)?;
            match shape {
                LoweredTensorShape::Unranked => format!("tensor<*x{element}>"),
                LoweredTensorShape::Ranked(dimensions) => {
                    let dimensions = dimensions
                        .iter()
                        .map(|dimension| match dimension {
                            LoweredTensorDimension::Dynamic => "?".into(),
                            LoweredTensorDimension::Known(value) => value.to_string(),
                        })
                        .collect::<Vec<_>>();
                    if dimensions.is_empty() {
                        format!("tensor<{element}>")
                    } else {
                        format!("tensor<{}x{element}>", dimensions.join("x"))
                    }
                }
            }
        }
        LoweredType::Task(_) => {
            let result = ty
                .clone()
                .task_result()
                .expect("the task variant always has a result type");
            if result == LoweredType::Unit {
                "!async.token".into()
            } else {
                format!("!async.value<{}>", mlir_type(&result)?)
            }
        }
    })
}

fn tensor_aggregate_abi_type(ty: &LoweredType) -> Option<&'static str> {
    matches!(
        ty,
        LoweredType::Tensor {
            shape: LoweredTensorShape::Unranked,
            ..
        }
    )
    .then_some("!llvm.ptr")
}

const UNRANKED_MEMREF_DESCRIPTOR_TYPE: &str = "!llvm.struct<(i64, !llvm.ptr)>";

fn render_tensor_aggregate_box(
    output: &mut String,
    source: &str,
    source_type: &LoweredType,
    tensor_type: &LoweredType,
    result: &str,
    unique: &str,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let unranked_tensor = unranked_tensor_type(tensor_type)?;
    let unranked_memref = unranked_memref_type(tensor_type)?;
    let tensor = if matches!(
        source_type,
        LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(_),
            ..
        }
    ) {
        let cast = format!("%{unique}_unranked");
        output.push_str(&format!(
            "{indentation}{cast} = tensor.cast {source} : {} to {unranked_tensor}\n",
            mlir_type(source_type)?
        ));
        cast
    } else {
        source.to_owned()
    };
    let buffer = format!("%{unique}_buffer");
    let descriptor = format!("%{unique}_descriptor");
    let rank = format!("%{unique}_rank");
    let pointer = format!("%{unique}_pointer");
    output.push_str(&format!(
        "{indentation}{buffer} = bufferization.to_buffer {tensor} read_only : {unranked_tensor} to {unranked_memref}\n"
    ));
    output.push_str(&format!(
        "{indentation}{descriptor} = builtin.unrealized_conversion_cast {buffer} : {unranked_memref} to {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{rank} = llvm.extractvalue {descriptor}[0] : {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{pointer} = llvm.extractvalue {descriptor}[1] : {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{result} = func.call @__sev_tensor_box_unranked({rank}, {pointer}) : (i64, !llvm.ptr) -> !llvm.ptr\n"
    ));
    Ok(())
}

fn render_tensor_aggregate_unbox(
    output: &mut String,
    boxed: &str,
    tensor_type: &LoweredType,
    result_type: &LoweredType,
    result: &str,
    unique: &str,
    indent: usize,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    let unranked_tensor = unranked_tensor_type(tensor_type)?;
    let unranked_memref = unranked_memref_type(tensor_type)?;
    let rank = format!("%{unique}_rank");
    let pointer = format!("%{unique}_pointer");
    let descriptor0 = format!("%{unique}_descriptor0");
    let descriptor1 = format!("%{unique}_descriptor1");
    let descriptor = format!("%{unique}_descriptor");
    let buffer = format!("%{unique}_buffer");
    output.push_str(&format!(
        "{indentation}{rank} = func.call @__sev_tensor_box_rank({boxed}) : (!llvm.ptr) -> i64\n"
    ));
    output.push_str(&format!(
        "{indentation}{pointer} = func.call @__sev_tensor_box_descriptor({boxed}) : (!llvm.ptr) -> !llvm.ptr\n"
    ));
    output.push_str(&format!(
        "{indentation}{descriptor0} = llvm.mlir.undef : {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{descriptor1} = llvm.insertvalue {rank}, {descriptor0}[0] : {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{descriptor} = llvm.insertvalue {pointer}, {descriptor1}[1] : {UNRANKED_MEMREF_DESCRIPTOR_TYPE}\n"
    ));
    output.push_str(&format!(
        "{indentation}{buffer} = builtin.unrealized_conversion_cast {descriptor} : {UNRANKED_MEMREF_DESCRIPTOR_TYPE} to {unranked_memref}\n"
    ));
    if matches!(
        result_type,
        LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(_),
            ..
        }
    ) {
        let tensor = format!("%{unique}_unranked");
        output.push_str(&format!(
            "{indentation}{tensor} = bufferization.to_tensor {buffer} restrict writable : {unranked_memref} to {unranked_tensor}\n"
        ));
        output.push_str(&format!(
            "{indentation}{result} = tensor.cast {tensor} : {unranked_tensor} to {}\n",
            mlir_type(result_type)?
        ));
    } else {
        output.push_str(&format!(
            "{indentation}{result} = bufferization.to_tensor {buffer} restrict writable : {unranked_memref} to {unranked_tensor}\n"
        ));
    }
    Ok(())
}

fn artifact_parameter(index: usize, ty: &LoweredType) -> Result<String, MlirError> {
    let spelling = mlir_type(ty)?;
    if matches!(ty, LoweredType::Tensor { .. }) {
        Ok(format!(
            "%arg{index}: {spelling} {{bufferization.access = \"read\"}}"
        ))
    } else {
        Ok(format!("%arg{index}: {spelling}"))
    }
}

fn aggregate_field_type(ty: &LoweredType) -> Result<String, MlirError> {
    tensor_aggregate_abi_type(ty)
        .map(str::to_owned)
        .map_or_else(|| mlir_type(ty), Ok)
}

fn unranked_tensor_type(ty: &LoweredType) -> Result<String, MlirError> {
    let LoweredType::Tensor { element, .. } = ty else {
        return Err(MlirError::UnsupportedOperation(format!(
            "aggregate field {ty:?} is not a tensor"
        )));
    };
    Ok(format!("tensor<*x{}>", mlir_tensor_element(*element)?))
}

fn unranked_memref_type(ty: &LoweredType) -> Result<String, MlirError> {
    let LoweredType::Tensor { element, .. } = ty else {
        return Err(MlirError::UnsupportedOperation(format!(
            "aggregate field {ty:?} is not a tensor"
        )));
    };
    Ok(format!("memref<*x{}>", mlir_tensor_element(*element)?))
}

fn aggregate_declaration_field_type<'a>(
    module: &'a Module,
    aggregate: &LoweredType,
    field: usize,
) -> Result<&'a LoweredType, MlirError> {
    let LoweredType::Aggregate(id) = aggregate else {
        return Err(MlirError::UnsupportedOperation(format!(
            "projected value {aggregate:?} is not an aggregate"
        )));
    };
    module
        .classes
        .iter()
        .find(|class| class.id == *id)
        .and_then(|class| class.fields.get(field))
        .map(|field| &field.ty)
        .ok_or_else(|| {
            MlirError::UnsupportedOperation(format!("aggregate {id} has no field {field}"))
        })
}

fn cfg_aggregate_field_type<'a>(
    module: &'a Module,
    body: &severian_lir::CfgBody,
    place: &severian_lir::Place,
    field: u32,
) -> Result<&'a LoweredType, MlirError> {
    let base = cfg_place_base_type(module, body, place)?;
    aggregate_declaration_field_type(module, &base, field as usize)
}

fn mlir_tensor_element(element: LoweredTensorElement) -> Result<String, MlirError> {
    mlir_type(&match element {
        LoweredTensorElement::Integer { bits, signed } => LoweredType::Integer { bits, signed },
        LoweredTensorElement::Float { format } => LoweredType::Float { format },
        LoweredTensorElement::Boolean => LoweredType::Boolean,
    })
}

fn mlir_binary(operator: BinaryOperation, ty: &LoweredType) -> Result<&'static str, MlirError> {
    let float = matches!(ty, LoweredType::Float { .. });
    let signed = matches!(ty, LoweredType::Integer { signed: true, .. });
    let integer = matches!(ty, LoweredType::Integer { .. } | LoweredType::Boolean);
    Ok(match (operator, float, integer) {
        (BinaryOperation::BitwiseOr | BinaryOperation::Or, false, true) => "arith.ori",
        (BinaryOperation::BitwiseAnd | BinaryOperation::And, false, true) => "arith.andi",
        (BinaryOperation::BitwiseXor, false, true) => "arith.xori",
        (BinaryOperation::Add, false, true) => "arith.addi",
        (BinaryOperation::Subtract, false, true) => "arith.subi",
        (BinaryOperation::Multiply, false, true) => "arith.muli",
        (BinaryOperation::Divide, false, true) if signed => "arith.divsi",
        (BinaryOperation::Divide, false, true) => "arith.divui",
        (BinaryOperation::FloorDivide, false, true) if signed => "arith.floordivsi",
        (BinaryOperation::FloorDivide, false, true) => "arith.divui",
        (BinaryOperation::Remainder, false, true) if signed => "arith.remsi",
        (BinaryOperation::Remainder, false, true) => "arith.remui",
        (BinaryOperation::Power, false, true) => "math.ipowi",
        (BinaryOperation::ShiftLeft, false, true) => "arith.shli",
        (BinaryOperation::ShiftRight, false, true) if signed => "arith.shrsi",
        (BinaryOperation::ShiftRight, false, true) => "arith.shrui",
        (BinaryOperation::Equal, false, true) => "arith.cmpi eq,",
        (BinaryOperation::NotEqual, false, true) => "arith.cmpi ne,",
        (BinaryOperation::Less, false, true) if signed => "arith.cmpi slt,",
        (BinaryOperation::Less, false, true) => "arith.cmpi ult,",
        (BinaryOperation::LessEqual, false, true) if signed => "arith.cmpi sle,",
        (BinaryOperation::LessEqual, false, true) => "arith.cmpi ule,",
        (BinaryOperation::Greater, false, true) if signed => "arith.cmpi sgt,",
        (BinaryOperation::Greater, false, true) => "arith.cmpi ugt,",
        (BinaryOperation::GreaterEqual, false, true) if signed => "arith.cmpi sge,",
        (BinaryOperation::GreaterEqual, false, true) => "arith.cmpi uge,",
        (BinaryOperation::Add, true, false) => "arith.addf",
        (BinaryOperation::Subtract, true, false) => "arith.subf",
        (BinaryOperation::Multiply, true, false) => "arith.mulf",
        (BinaryOperation::Divide, true, false) => "arith.divf",
        (BinaryOperation::Remainder, true, false) => "arith.remf",
        (BinaryOperation::Equal, true, false) => "arith.cmpf oeq,",
        (BinaryOperation::NotEqual, true, false) => "arith.cmpf une,",
        (BinaryOperation::Less, true, false) => "arith.cmpf olt,",
        (BinaryOperation::LessEqual, true, false) => "arith.cmpf ole,",
        (BinaryOperation::Greater, true, false) => "arith.cmpf ogt,",
        (BinaryOperation::GreaterEqual, true, false) => "arith.cmpf oge,",
        _ => {
            return Err(MlirError::UnsupportedOperation(format!(
                "MLIR binary lowering is unavailable for {operator:?} on {ty:?}"
            )))
        }
    })
}

#[allow(dead_code)]
fn _unary_is_lir(_: UnaryOperation) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compose, compose_gpu_launchers, verify_artifact, GpuLaunchArtifact};
    use severian_artifact::CompiledRegionId;
    use severian_target::TargetSpec;

    fn artifact_id() -> ArtifactId {
        ArtifactId::for_region(CompiledRegionId::new(0))
    }

    #[test]
    fn bfloat_is_not_silently_mapped_to_f32() {
        assert_eq!(
            mlir_type(&LoweredType::Float {
                format: LoweredFloatFormat::BrainFloat16
            })
            .unwrap(),
            "bf16"
        );
    }

    #[test]
    fn tensor_float_widths_keep_their_physical_mlir_spelling() {
        let cases = [
            (LoweredFloatFormat::Float8E4M3Fn, "i8"),
            (LoweredFloatFormat::Float8E5M2, "i8"),
            (LoweredFloatFormat::Ieee(16), "f16"),
            (LoweredFloatFormat::BrainFloat16, "bf16"),
            (LoweredFloatFormat::Ieee(32), "f32"),
            (LoweredFloatFormat::Ieee(64), "f64"),
            (LoweredFloatFormat::Ieee(80), "f80"),
            (LoweredFloatFormat::Ieee(128), "f128"),
        ];
        for (format, expected) in cases {
            assert_eq!(mlir_type(&LoweredType::Float { format }).unwrap(), expected);
        }
    }

    #[test]
    fn extended_tensor_float_widths_verify_at_the_mlir_boundary() {
        for (format, spelling) in [
            (LoweredFloatFormat::Float8E4M3Fn, "i8"),
            (LoweredFloatFormat::Float8E5M2, "i8"),
            (LoweredFloatFormat::Ieee(80), "f80"),
            (LoweredFloatFormat::Ieee(128), "f128"),
        ] {
            let ty = LoweredType::Float { format };
            verify_artifact(
                artifact_id(),
                MlirArtifact {
                    module: format!(
                        "module {{ func.func @entry(%arg0: {spelling}) -> {spelling} {{ return %arg0 : {spelling} }} }}"
                    ),
                    inputs: vec![ty.clone()],
                    outputs: vec![ty],
                },
                &TargetSpec::host(),
            )
            .unwrap();
        }
    }

    #[test]
    fn byte_storage_uses_the_runtime_pointer_representation() {
        assert_eq!(mlir_type(&LoweredType::Bytes).unwrap(), "!llvm.ptr");
    }

    #[test]
    fn aggregate_aliases_are_emitted_after_their_field_dependencies() {
        let module = Module {
            classes: vec![
                severian_lir::ClassDeclaration {
                    id: 3,
                    name: "Outer".into(),
                    fields: vec![severian_lir::ClassFieldDeclaration {
                        name: "inner".into(),
                        ty: LoweredType::Aggregate(22),
                    }],
                },
                severian_lir::ClassDeclaration {
                    id: 22,
                    name: "Inner".into(),
                    fields: vec![severian_lir::ClassFieldDeclaration {
                        name: "value".into(),
                        ty: LoweredType::Integer {
                            bits: 64,
                            signed: true,
                        },
                    }],
                },
            ],
            ..Module::default()
        };
        let mut output = String::new();
        render_class_aliases(&mut output, &module).unwrap();
        assert!(output.find("!sev_class_22 =").unwrap() < output.find("!sev_class_3 =").unwrap());
    }

    #[test]
    fn gpu_cfg_placement_becomes_a_gpu_launch_region() {
        let integer = LoweredType::Integer {
            bits: 64,
            signed: true,
        };
        let module = Module {
            values: (0..3)
                .map(|id| severian_lir::Value {
                    id: ValueId(id),
                    ty: integer.clone(),
                })
                .collect(),
            initializer_cfg: Some(severian_lir::CfgBody {
                entry: severian_lir::BlockId(0),
                blocks: vec![
                    severian_lir::BasicBlock {
                        id: severian_lir::BlockId(0),
                        execution: None,
                        operations: Vec::new(),
                        operation_spans: Vec::new(),
                        terminator: severian_lir::Terminator::Goto(severian_lir::BlockId(1)),
                        terminator_span: None,
                    },
                    severian_lir::BasicBlock {
                        id: severian_lir::BlockId(1),
                        execution: Some(severian_universal::ExecutionPlacement::Gpu),
                        operations: vec![
                            Operation::Constant {
                                value: Constant::Integer("20".into()),
                                result: ValueId(0),
                            },
                            Operation::Constant {
                                value: Constant::Integer("22".into()),
                                result: ValueId(1),
                            },
                            Operation::Binary {
                                operator: BinaryOperation::Add,
                                left: ValueId(0),
                                right: ValueId(1),
                                result: ValueId(2),
                            },
                        ],
                        operation_spans: vec![None; 3],
                        terminator: severian_lir::Terminator::Goto(severian_lir::BlockId(2)),
                        terminator_span: None,
                    },
                    severian_lir::BasicBlock {
                        id: severian_lir::BlockId(2),
                        execution: None,
                        operations: Vec::new(),
                        operation_spans: Vec::new(),
                        terminator: severian_lir::Terminator::Return(None),
                        terminator_span: None,
                    },
                ],
                locals: Vec::new(),
                return_type: LoweredType::Unit,
            }),
            ..Module::default()
        };

        let rendered = render(&module).unwrap();
        assert!(rendered.contains("gpu.launch blocks"));
        assert!(rendered.contains("arith.addi %v0, %v1"));
        assert!(rendered.contains("^gpu1_exit:"));
        assert!(!rendered.contains("\n  ^bb1:"));
        assert!(rendered.contains("cf.br ^bb2"));
        let mut target = TargetSpec::host();
        target.capabilities.insert("mlir.dialect.gpu");
        compose(&rendered, &[], &target).unwrap();
    }

    #[test]
    fn strings_use_llvm_globals_without_changing_their_bytes() {
        assert_eq!(mlir_string("a\n\"b\\é"), "a\\0A\\22b\\5C\\C3\\A9");
    }

    #[test]
    fn repeated_coverage_points_share_a_global_but_use_distinct_ssa_values() {
        let key = "example.sev|statement|7|12|0";
        let module = Module {
            initializer: Block {
                operations: vec![
                    Operation::Coverage { key: key.into() },
                    Operation::Coverage { key: key.into() },
                ],
            },
            ..Module::default()
        };
        let rendered = render(&module).unwrap();
        let symbol = coverage_symbol(key);
        assert_eq!(rendered.matches(&format!("@{symbol}(\"")).count(), 1);
        assert!(rendered.contains(&format!("%{symbol}_0 = llvm.mlir.addressof")));
        assert!(rendered.contains(&format!("%{symbol}_1 = llvm.mlir.addressof")));
    }

    #[test]
    fn tasks_render_with_the_async_dialect_and_preserve_policy() {
        let integer = LoweredType::Integer {
            bits: 64,
            signed: true,
        };
        let module = Module {
            values: vec![
                severian_lir::Value {
                    id: ValueId(0),
                    ty: integer.clone(),
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: integer.clone(),
                },
                severian_lir::Value {
                    id: ValueId(2),
                    ty: integer.clone(),
                },
                severian_lir::Value {
                    id: ValueId(3),
                    ty: integer.clone(),
                },
            ],
            functions: vec![
                Function {
                    id: FunctionId(1),
                    name: "work".into(),
                    parameters: vec![ValueId(3)],
                    result: integer.clone(),
                    body: Some(Block {
                        operations: vec![Operation::Return {
                            value: Some(ValueId(3)),
                        }],
                    }),
                    linkage: FunctionLinkage::Internal,
                    parameter_types: vec![integer.clone()],
                    cfg: None,
                },
                Function {
                    id: FunctionId(2),
                    name: "main".into(),
                    parameters: vec![],
                    result: LoweredType::Unit,
                    body: Some(Block {
                        operations: vec![
                            Operation::Constant {
                                value: Constant::Integer("21".into()),
                                result: ValueId(0),
                            },
                            Operation::Spawn {
                                function: FunctionId(1),
                                arguments: vec![ValueId(0)],
                                result: ValueId(1),
                                owner: severian_lir::TaskOwner::SelfScope,
                                locked: true,
                            },
                            Operation::Await {
                                task: ValueId(1),
                                result: ValueId(2),
                            },
                            Operation::Return { value: None },
                        ],
                    }),
                    linkage: FunctionLinkage::Internal,
                    parameter_types: vec![],
                    cfg: None,
                },
            ],
            entry: Some(FunctionId(2)),
            ..Module::default()
        };
        let rendered = render(&module).unwrap();
        assert!(rendered.contains("severian.task owner=self locked=true"));
        assert!(rendered.contains("async.execute"));
        assert!(rendered.contains("async.await"));
        assert!(rendered.contains("func.call @__sev_task_lock()"));
        assert!(rendered.contains("func.call @__sev_task_unlock()"));
    }

    #[test]
    fn mlir_emitter_handles_lowered_runtime_calls_generically() {
        let module = Module {
            values: (0..3)
                .map(|id| severian_lir::Value {
                    id: ValueId(id),
                    ty: LoweredType::String,
                })
                .collect(),
            initializer: Block {
                operations: vec![
                    Operation::Constant {
                        value: Constant::String("left".into()),
                        result: ValueId(0),
                    },
                    Operation::Constant {
                        value: Constant::String("right".into()),
                        result: ValueId(1),
                    },
                    Operation::RuntimeCall {
                        symbol: "__sev_string_concat".into(),
                        arguments: vec![ValueId(0), ValueId(1)],
                        result: Some(ValueId(2)),
                    },
                ],
            },
            ..Module::default()
        };
        let rendered = render(&module).unwrap();
        assert!(rendered.contains("func.call @__sev_string_concat(%v0, %v1)"));
    }

    #[test]
    fn native_runtime_calls_cannot_disguise_storage_pointers_as_tensors() {
        let module = Module {
            values: vec![severian_lir::Value {
                id: ValueId(0),
                ty: LoweredType::Tensor {
                    element: LoweredTensorElement::Float {
                        format: LoweredFloatFormat::BrainFloat16,
                    },
                    shape: LoweredTensorShape::Unranked,
                },
            }],
            initializer: Block {
                operations: vec![Operation::RuntimeCall {
                    symbol: "__sev_safetensor_view".into(),
                    arguments: Vec::new(),
                    result: Some(ValueId(0)),
                }],
            },
            ..Module::default()
        };
        let error = render(&module).unwrap_err().to_string();
        assert!(error.contains("cannot receive or return an MLIR builtin tensor"));
        assert!(error.contains("!llvm.ptr StorageViewAbi"));
        assert!(!error.contains("tensor<*xbf16>"));
    }

    #[test]
    fn safetensor_loader_has_one_pointer_returning_mlir_declaration() {
        let module = Module {
            values: vec![
                severian_lir::Value {
                    id: ValueId(0),
                    ty: LoweredType::Integer {
                        bits: 64,
                        signed: true,
                    },
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: LoweredType::String,
                },
                severian_lir::Value {
                    id: ValueId(2),
                    ty: LoweredType::Bytes,
                },
            ],
            initializer: Block {
                operations: vec![
                    Operation::Constant {
                        value: Constant::Integer("1".into()),
                        result: ValueId(0),
                    },
                    Operation::Constant {
                        value: Constant::String("weight".into()),
                        result: ValueId(1),
                    },
                    Operation::RuntimeCall {
                        symbol: "__sev_safetensor_view".into(),
                        arguments: vec![ValueId(0), ValueId(1)],
                        result: Some(ValueId(2)),
                    },
                ],
            },
            ..Module::default()
        };
        let rendered = render(&module).unwrap();
        assert!(rendered
            .contains("func.func private @__sev_safetensor_view(i64, !llvm.ptr) -> !llvm.ptr"));
        assert!(!rendered.contains("@__sev_safetensor_view(i64, !llvm.ptr) -> tensor<"));
        assert!(!rendered.contains("__sev_safetensor_bf16_view"));
    }

    #[test]
    fn integer_comparisons_preserve_signedness() {
        assert_eq!(
            mlir_binary(
                BinaryOperation::Equal,
                &LoweredType::Integer {
                    bits: 32,
                    signed: false,
                },
            )
            .unwrap(),
            "arith.cmpi eq,"
        );
        assert_eq!(
            mlir_binary(
                BinaryOperation::Less,
                &LoweredType::Integer {
                    bits: 32,
                    signed: false,
                },
            )
            .unwrap(),
            "arith.cmpi ult,"
        );
    }

    #[test]
    fn numeric_conversions_emit_real_arithmetic_casts() {
        let integer = LoweredType::Integer {
            bits: 64,
            signed: true,
        };
        let float = LoweredType::Float {
            format: LoweredFloatFormat::Ieee(64),
        };
        let rendered = render(&Module {
            values: vec![
                severian_lir::Value {
                    id: ValueId(0),
                    ty: integer,
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: float,
                },
            ],
            initializer: Block {
                operations: vec![
                    Operation::Constant {
                        value: Constant::Integer("10".into()),
                        result: ValueId(0),
                    },
                    Operation::Convert {
                        operand: ValueId(0),
                        result: ValueId(1),
                        kind: severian_universal::ConversionKind::Lossy,
                    },
                ],
            },
            ..Module::default()
        })
        .unwrap();
        assert!(rendered.contains("arith.sitofp %v0 : i64 to f64"));
    }

    #[test]
    fn leading_dot_float_literals_are_normalized_for_mlir() {
        assert_eq!(mlir_float_literal(".5"), "0.5");
        assert_eq!(mlir_float_literal("1.5"), "1.5");
        assert_eq!(mlir_float_literal("1e-8"), "1.0e-8");
        assert_eq!(mlir_float_literal("2.5E+10"), "2.5E+10");
    }

    #[test]
    fn cfg_boolean_not_uses_an_ssa_constant_operand() {
        let boolean = LoweredType::Boolean;
        let module = Module {
            values: vec![
                severian_lir::Value {
                    id: ValueId(0),
                    ty: boolean.clone(),
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: boolean.clone(),
                },
            ],
            ..Module::default()
        };
        let mut rendered = String::new();
        render_cfg_unary(
            &mut rendered,
            &module,
            UnaryOperation::Not,
            ValueId(0),
            ValueId(1),
            4,
        )
        .unwrap();
        assert!(rendered.contains("%v1_not = arith.constant true"));
        assert!(rendered.contains("arith.xori %v0, %v1_not : i1"));
    }

    #[test]
    fn generated_artifacts_are_verified_and_composed() {
        let target = TargetSpec::new("x86_64-unknown-linux");
        let artifact = MlirArtifact {
            module: "module {\n  func.func @local_entry() {\n    return\n  }\n}".into(),
            inputs: vec![],
            outputs: vec![],
        };
        let artifact = verify_artifact(artifact_id(), artifact, &target).unwrap();
        let composed = compose("module {\n}\n", &[artifact], &target).unwrap();
        assert!(composed.contains("func.func @__sev_artifact_0"));
    }

    #[test]
    fn artifact_calls_and_composition_support_multiple_results() {
        let target = TargetSpec::new("x86_64-unknown-linux");
        let i32_type = LoweredType::Integer {
            bits: 32,
            signed: true,
        };
        let f32_type = LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        };
        let ordinary = render(&Module {
            values: vec![
                severian_lir::Value {
                    id: ValueId(0),
                    ty: i32_type.clone(),
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: f32_type.clone(),
                },
            ],
            globals: vec![],
            initializer: severian_lir::Block {
                operations: vec![Operation::ArtifactCall {
                    artifact: artifact_id(),
                    inputs: vec![],
                    outputs: vec![ValueId(0), ValueId(1)],
                }],
            },
            functions: vec![],
            entry: None,
            traits: vec![],
            classes: vec![],
            storage_globals: vec![],
            initializer_cfg: None,
            gpu_architecture: None,
        })
        .unwrap();
        let artifact = verify_artifact(
            artifact_id(),
            MlirArtifact {
                module: "module { func.func @entry() -> (i32, f32) { %0 = arith.constant 1 : i32 %1 = arith.constant 2.0 : f32 return %0, %1 : i32, f32 } }".into(),
                inputs: vec![],
                outputs: vec![i32_type, f32_type],
            },
            &target,
        )
        .unwrap();
        let composed = compose(&ordinary, &[artifact], &target).unwrap();
        assert!(composed.contains("call @__sev_artifact_0"));
        assert!(composed.contains("func.func @__sev_artifact_0"));
    }

    #[test]
    fn nested_artifact_calls_are_declared_and_rendered_inside_control_flow() {
        let ordinary = render(&Module {
            values: vec![severian_lir::Value {
                id: ValueId(0),
                ty: LoweredType::Boolean,
            }],
            globals: vec![],
            initializer: Block {
                operations: vec![
                    Operation::Constant {
                        value: Constant::Boolean(true),
                        result: ValueId(0),
                    },
                    Operation::If {
                        condition: ValueId(0),
                        then_block: Block {
                            operations: vec![Operation::ArtifactCall {
                                artifact: artifact_id(),
                                inputs: vec![],
                                outputs: vec![],
                            }],
                        },
                        else_block: Block::default(),
                    },
                ],
            },
            functions: vec![],
            entry: None,
            traits: vec![],
            classes: vec![],
            storage_globals: vec![],
            initializer_cfg: None,
            gpu_architecture: None,
        })
        .unwrap();

        assert!(ordinary.contains("func.func private @__sev_artifact_0()"));
        assert!(ordinary.contains("scf.if %v0"));
        assert!(ordinary.contains("func.call @__sev_artifact_0()"));
        compose(&ordinary, &[], &TargetSpec::new("x86_64-unknown-linux")).unwrap();
    }

    #[test]
    fn gpu_artifact_calls_are_backed_only_by_host_launcher_functions() {
        let target = TargetSpec::new("x86_64-unknown-linux");
        let tensor = LoweredType::Tensor {
            element: LoweredTensorElement::Float {
                format: LoweredFloatFormat::Ieee(32),
            },
            shape: LoweredTensorShape::Unranked,
        };
        let ordinary = "module { func.func private @__sev_artifact_0(tensor<*xf32>) -> tensor<*xf32> func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> { %0 = func.call @__sev_artifact_0(%arg0) : (tensor<*xf32>) -> tensor<*xf32> return %0 : tensor<*xf32> } }";
        let composed = compose_gpu_launchers(
            &ordinary,
            &[GpuLaunchArtifact {
                id: artifact_id(),
                inputs: vec![tensor.clone()],
                outputs: vec![tensor],
            }],
            &target,
        )
        .unwrap();

        assert!(composed.contains("call @__sev_gpu_launch_0"), "{composed}");
        assert!(
            composed.contains("func.func private @__sev_gpu_launch_0"),
            "{composed}"
        );
        assert!(
            composed.contains("func.func private @__sev_artifact_0"),
            "{composed}"
        );
        assert!(!composed.contains("__sev_tensor_op_"));
    }

    #[test]
    fn invalid_or_disallowed_generated_ir_is_rejected() {
        let target = TargetSpec::new("x86_64-unknown-linux");
        let helper = MlirArtifact {
            module: "module { func.func private @helper() { return } func.func @entry() { func.call @helper() : () -> () return } }".into(),
            inputs: vec![],
            outputs: vec![],
        };
        verify_artifact(artifact_id(), helper, &target).unwrap();

        let invalid = MlirArtifact {
            module: "module { func.func @bad() {".into(),
            inputs: vec![],
            outputs: vec![],
        };
        assert!(matches!(
            verify_artifact(artifact_id(), invalid, &target),
            Err(MlirError::ParseFailed(_))
        ));

        let declaration = MlirArtifact {
            module: "module { func.func private @declaration() }".into(),
            inputs: vec![],
            outputs: vec![],
        };
        assert!(matches!(
            verify_artifact(artifact_id(), declaration, &target),
            Err(MlirError::EntryFunctionIsDeclaration)
        ));

        let wrong_signature = MlirArtifact {
            module: "module {\n  func.func @wrong_signature(%arg0: i64) {\n    return\n  }\n}"
                .into(),
            inputs: vec![LoweredType::Integer {
                bits: 32,
                signed: true,
            }],
            outputs: vec![],
        };
        assert!(matches!(
            verify_artifact(artifact_id(), wrong_signature, &target),
            Err(MlirError::SignatureMismatch)
        ));

        let disallowed = MlirArtifact {
            module: "module {\n  func.func @vector_entry(%arg0: f32) -> f32 {\n    %0 = vector.broadcast %arg0 : f32 to vector<1xf32>\n    %1 = vector.extract %0[0] : f32 from vector<1xf32>\n    return %1 : f32\n  }\n}".into(),
            inputs: vec![LoweredType::Float {
                format: LoweredFloatFormat::Ieee(32),
            }],
            outputs: vec![LoweredType::Float {
                format: LoweredFloatFormat::Ieee(32),
            }],
        };
        assert!(matches!(
            verify_artifact(artifact_id(), disallowed, &target),
            Err(MlirError::DialectNotAllowed { .. })
        ));
    }
}
fn render_class_aliases(output: &mut String, module: &Module) -> Result<(), MlirError> {
    let declared = module
        .classes
        .iter()
        .map(|declaration| declaration.id)
        .collect::<BTreeSet<_>>();
    let mut emitted = BTreeSet::new();
    while emitted.len() != module.classes.len() {
        let before = emitted.len();
        for declaration in &module.classes {
            if emitted.contains(&declaration.id) {
                continue;
            }
            let ready = declaration.fields.iter().all(|field| match field.ty {
                LoweredType::Aggregate(dependency) => {
                    !declared.contains(&dependency) || emitted.contains(&dependency)
                }
                _ => true,
            });
            if !ready {
                continue;
            }
            let fields = declaration
                .fields
                .iter()
                .map(|field| aggregate_field_type(&field.ty))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            output.push_str(&format!(
                "!sev_class_{} = !llvm.struct<({fields})>\n",
                declaration.id
            ));
            emitted.insert(declaration.id);
        }
        if emitted.len() == before {
            let cyclic = declared
                .difference(&emitted)
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(MlirError::UnsupportedOperation(format!(
                "aggregate layout contains an inline cycle among class IDs {cyclic}"
            )));
        }
    }
    Ok(())
}
