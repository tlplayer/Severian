use severian_artifact::ArtifactId;
use severian_lir::{
    BinaryOperation, Block, Constant, Function, FunctionId, FunctionLinkage, LoweredFloatFormat,
    LoweredType, Module, Operation, UnaryOperation, ValueId,
};
use std::collections::BTreeMap;
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
    let mut artifact_signatures =
        BTreeMap::<ArtifactId, (Vec<LoweredType>, Vec<LoweredType>)>::new();
    for operation in all_operations(module) {
        if let Operation::ArtifactCall {
            artifact,
            inputs,
            outputs,
        } = operation
        {
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
    }
    let coverage = all_operations(module)
        .into_iter()
        .filter_map(|operation| match operation {
            Operation::Coverage { key } => Some(key),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut output = String::from("module {\n");
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
            .map(mlir_type)
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let outputs = outputs
            .into_iter()
            .map(mlir_type)
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
        render_function_declaration(&mut output, module, function)?;
    }
    for function in module
        .functions
        .iter()
        .filter(|function| function.body.is_some())
    {
        render_function_definition(&mut output, module, function)?;
    }
    output.push_str("  func.func @main() -> i32 {\n");
    render_block(&mut output, module, &module.initializer, 4, None)?;
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

fn render_block(
    output: &mut String,
    module: &Module,
    block: &Block,
    indent: usize,
    function_result: Option<LoweredType>,
) -> Result<(), MlirError> {
    let indentation = " ".repeat(indent);
    for operation in &block.operations {
        match operation {
            Operation::Coverage { key } => {
                let symbol = coverage_symbol(key);
                output.push_str(&format!(
                    "{indentation}%{symbol} = llvm.mlir.addressof @{symbol} : !llvm.ptr\n"
                ));
                output.push_str(&format!(
                    "{indentation}func.call @__sev_coverage_hit(%{symbol}) : (!llvm.ptr) -> ()\n"
                ));
            }
            Operation::Constant { value, result } => {
                let ty = value_type(module, *result)?;
                let spelling = mlir_type(ty)?;
                let literal = match value {
                    Constant::Integer(value) | Constant::Float(value) => value,
                    Constant::Boolean(true) => "1",
                    Constant::Boolean(false) => "0",
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
                let ty = mlir_type(value_type(module, *result)?)?;
                return Err(MlirError::UnsupportedOperation(format!(
                    "MLIR unary {operator:?} for %v{} -> %v{} : {ty} requires a dedicated lowering",
                    operand.0, result.0
                )));
            }
            Operation::Binary {
                operator,
                left,
                right,
                result,
            } => {
                let input_type = value_type(module, *left)?;
                let spelling = mlir_type(input_type)?;
                let instruction = mlir_binary(*operator, input_type)?;
                output.push_str(&format!(
                    "{indentation}%v{} = {instruction} %v{}, %v{} : {spelling}\n",
                    result.0, left.0, right.0
                ));
            }
            Operation::Call {
                function: target,
                arguments,
                result,
            } => {
                let target = function(module, *target)?;
                let arguments = arguments
                    .iter()
                    .map(|value| format!("%v{}", value.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let argument_types = argument_types(module, target)?;
                if target.result == LoweredType::Unit {
                    output.push_str(&format!(
                        "{indentation}func.call @{}({arguments}) : ({argument_types}) -> ()\n",
                        function_symbol(target),
                    ));
                } else {
                    let result_type = mlir_type(value_type(module, *result)?)?;
                    output.push_str(&format!(
                        "{indentation}%v{} = func.call @{}({arguments}) : ({argument_types}) -> {result_type}\n",
                        result.0,
                        function_symbol(target),
                    ));
                }
            }
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
                    (Some(value), expected) if expected != LoweredType::Unit => {
                        let actual = value_type(module, *value)?;
                        if actual != expected {
                            return Err(MlirError::SignatureMismatch);
                        }
                        output.push_str(&format!(
                            "{indentation}return %v{} : {}\n",
                            value.0,
                            mlir_type(actual)?
                        ));
                    }
                    _ => return Err(MlirError::SignatureMismatch),
                }
            }
            Operation::Assert { .. } => {
                return Err(MlirError::UnsupportedOperation(
                    "assert requires the standard test runtime lowering path".into(),
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
                render_block(output, module, then_block, indent + 2, function_result)?;
                output.push_str(&format!("{}scf.yield\n", " ".repeat(indent + 2)));
                if else_block.operations.is_empty() {
                    output.push_str(&format!("{indentation}}}\n"));
                } else {
                    output.push_str(&format!("{indentation}}} else {{\n"));
                    render_block(output, module, else_block, indent + 2, function_result)?;
                    output.push_str(&format!("{}scf.yield\n", " ".repeat(indent + 2)));
                    output.push_str(&format!("{indentation}}}\n"));
                }
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
                    .map(|value| mlir_type(value_type(module, *value)?))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                match outputs.as_slice() {
                    [] => output.push_str(&format!(
                        "{indentation}func.call @{symbol}({arguments}) : ({input_types}) -> ()\n"
                    )),
                    [result] => {
                        let ty = mlir_type(value_type(module, *result)?)?;
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
                            .map(|result| mlir_type(value_type(module, *result)?))
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
        .map(|parameter| mlir_type(value_type(module, *parameter)?))
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
        .map(|parameter| mlir_type(value_type(module, *parameter)?))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let result = function_result(function.result)?;
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
                mlir_type(value_type(module, *parameter)?)?
            ))
        })
        .collect::<Result<Vec<_>, MlirError>>()?
        .join(", ");
    let result = function_result(function.result)?;
    output.push_str(&format!(
        "  func.func private @{}({parameters}){result} {{\n",
        function_symbol(function),
    ));
    let body = function.body.as_ref().expect("filtered source body");
    render_block(output, module, body, 4, Some(function.result))?;
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

fn function_result(result: LoweredType) -> Result<String, MlirError> {
    if result == LoweredType::Unit {
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
        .map(|value| value.ty)
        .ok_or(MlirError::InvalidValue(id))
}

pub(crate) fn mlir_type(ty: LoweredType) -> Result<String, MlirError> {
    Ok(match ty {
        LoweredType::Integer { bits, .. } => format!("i{bits}"),
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
            format: LoweredFloatFormat::BrainFloat16,
        } => "bf16".into(),
        LoweredType::Boolean => "i1".into(),
        LoweredType::String => "!llvm.ptr".into(),
        unsupported => return Err(MlirError::UnsupportedType(unsupported)),
    })
}

fn mlir_binary(operator: BinaryOperation, ty: LoweredType) -> Result<&'static str, MlirError> {
    let float = matches!(ty, LoweredType::Float { .. });
    let signed = matches!(ty, LoweredType::Integer { signed: true, .. });
    Ok(match (operator, float) {
        (BinaryOperation::Add, false) => "arith.addi",
        (BinaryOperation::Subtract, false) => "arith.subi",
        (BinaryOperation::Multiply, false) => "arith.muli",
        (BinaryOperation::Divide, false) if signed => "arith.divsi",
        (BinaryOperation::Divide, false) => "arith.divui",
        (BinaryOperation::Remainder, false) if signed => "arith.remsi",
        (BinaryOperation::Remainder, false) => "arith.remui",
        (BinaryOperation::Add, true) => "arith.addf",
        (BinaryOperation::Subtract, true) => "arith.subf",
        (BinaryOperation::Multiply, true) => "arith.mulf",
        (BinaryOperation::Divide, true) => "arith.divf",
        (BinaryOperation::Remainder, true) => "arith.remf",
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
    use crate::{compose, verify_artifact};
    use severian_artifact::CompiledRegionId;
    use severian_target::TargetSpec;

    fn artifact_id() -> ArtifactId {
        ArtifactId::for_region(CompiledRegionId::new(0))
    }

    #[test]
    fn bfloat_is_not_silently_mapped_to_f32() {
        assert_eq!(
            mlir_type(LoweredType::Float {
                format: LoweredFloatFormat::BrainFloat16
            })
            .unwrap(),
            "bf16"
        );
    }

    #[test]
    fn unsupported_aggregate_is_explicit() {
        assert!(matches!(
            mlir_type(LoweredType::Bytes),
            Err(MlirError::UnsupportedType(LoweredType::Bytes))
        ));
    }

    #[test]
    fn strings_use_llvm_globals_without_changing_their_bytes() {
        assert_eq!(mlir_string("a\n\"b\\é"), "a\\0A\\22b\\5C\\C3\\A9");
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
                    ty: i32_type,
                },
                severian_lir::Value {
                    id: ValueId(1),
                    ty: f32_type,
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
        })
        .unwrap();

        assert!(ordinary.contains("func.func private @__sev_artifact_0()"));
        assert!(ordinary.contains("scf.if %v0"));
        assert!(ordinary.contains("func.call @__sev_artifact_0()"));
        compose(&ordinary, &[], &TargetSpec::new("x86_64-unknown-linux")).unwrap();
    }

    #[test]
    fn invalid_or_disallowed_generated_ir_is_rejected() {
        let target = TargetSpec::new("x86_64-unknown-linux");
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
            module: "module {\n  func.func @math_entry(%arg0: f32) -> f32 {\n    %0 = math.absf %arg0 : f32\n    return %0 : f32\n  }\n}".into(),
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
