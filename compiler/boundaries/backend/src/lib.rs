#![forbid(unsafe_code)]

use severian_lir::LoweredFloatFormat;
pub use severian_lir::{
    BinaryOperation, Block, Constant, Function, FunctionId, FunctionLinkage, LoweredType,
    Module as LoweredModule, Operation, UnaryOperation, ValueId,
};
use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Executable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub path: PathBuf,
    pub kind: ArtifactKind,
}

#[derive(Debug)]
pub enum BackendError {
    UnsupportedType(LoweredType),
    UnsupportedOperation(String),
    InvalidValue(ValueId),
    Spawn(std::io::Error),
    Write(std::io::Error),
    Wait(std::io::Error),
    CompilerFailed(String),
    ToolSpawn {
        tool: &'static str,
        source: std::io::Error,
    },
    ToolWrite {
        tool: &'static str,
        source: std::io::Error,
    },
    ToolWait {
        tool: &'static str,
        source: std::io::Error,
    },
    ToolFailed {
        tool: &'static str,
        diagnostic: String,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedType(ty) => write!(formatter, "C backend does not support {ty:?}"),
            Self::UnsupportedOperation(message) => formatter.write_str(message),
            Self::InvalidValue(value) => write!(formatter, "invalid LIR value {}", value.0),
            Self::Spawn(error) => write!(formatter, "could not start native C compiler: {error}"),
            Self::Write(error) => write!(formatter, "could not write C source: {error}"),
            Self::Wait(error) => write!(formatter, "could not wait for native C compiler: {error}"),
            Self::CompilerFailed(error) => write!(formatter, "native C compiler failed: {error}"),
            Self::ToolSpawn { tool, source } => {
                write!(formatter, "could not start {tool}: {source}")
            }
            Self::ToolWrite { tool, source } => {
                write!(formatter, "could not write input to {tool}: {source}")
            }
            Self::ToolWait { tool, source } => {
                write!(formatter, "could not wait for {tool}: {source}")
            }
            Self::ToolFailed { tool, diagnostic } => {
                write!(formatter, "{tool} failed: {diagnostic}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

pub fn render_c(module: &LoweredModule) -> Result<String, BackendError> {
    let mut output = String::from(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n\ntypedef struct { int count; char **values; } sev_args;\n\nvoid __sev_coverage_hit(const char *key);\n",
    );
    let mut runtime_signatures = BTreeMap::new();
    for operation in all_operations(module) {
        let Operation::RuntimeCall {
            symbol,
            arguments,
            result,
        } = operation
        else {
            continue;
        };
        let inputs = arguments
            .iter()
            .map(|argument| value_type(module, *argument))
            .collect::<Result<Vec<_>, _>>()?;
        let output_type = result
            .map(|result| value_type(module, result))
            .transpose()?;
        if let Some(known) =
            runtime_signatures.insert(symbol.clone(), (inputs.clone(), output_type))
        {
            if known != (inputs, output_type) {
                return Err(BackendError::UnsupportedOperation(format!(
                    "runtime symbol `{symbol}` has conflicting physical signatures"
                )));
            }
        }
    }
    for (symbol, (inputs, result)) in runtime_signatures {
        let inputs = inputs
            .into_iter()
            .map(c_type)
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        output.push_str(&format!(
            "{} {symbol}({});\n",
            result.map(c_type).transpose()?.unwrap_or("void"),
            if inputs.is_empty() { "void" } else { &inputs }
        ));
    }
    output.push('\n');
    for value in &module.globals {
        output.push_str(&format!(
            "static {} v{};\n",
            c_type(value_type(module, *value)?)?,
            value.0
        ));
    }
    if !module.globals.is_empty() {
        output.push('\n');
    }
    for function in &module.functions {
        let parameters = function
            .parameters
            .iter()
            .map(|parameter| {
                Ok(format!(
                    "{} v{}",
                    c_type(value_type(module, *parameter)?)?,
                    parameter.0
                ))
            })
            .collect::<Result<Vec<_>, BackendError>>()?
            .join(", ");
        let name = function_name(function);
        let prefix = match function.linkage {
            FunctionLinkage::Internal => "static ",
            FunctionLinkage::External { .. } => "extern ",
        };
        output.push_str(&format!(
            "{prefix}{} {name}({});\n",
            c_return_type(function.result)?,
            if parameters.is_empty() {
                "void"
            } else {
                &parameters
            }
        ));
    }
    output.push_str("\nstatic int __sev_init_state;\nstatic int __sev_init(void) {\n    if (__sev_init_state == 2) return 0;\n    if (__sev_init_state == 1) return 1;\n    __sev_init_state = 1;\n");
    render_block(&mut output, module, &module.initializer)?;
    output.push_str("    __sev_init_state = 2;\n    return 0;\n}\n\n");

    for function in module
        .functions
        .iter()
        .filter(|function| function.body.is_some())
    {
        let parameters = function
            .parameters
            .iter()
            .map(|parameter| {
                Ok(format!(
                    "{} v{}",
                    c_type(value_type(module, *parameter)?)?,
                    parameter.0
                ))
            })
            .collect::<Result<Vec<_>, BackendError>>()?
            .join(", ");
        output.push_str(&format!(
            "static {} {}({}) {{\n",
            c_return_type(function.result)?,
            function_name(function),
            if parameters.is_empty() {
                "void"
            } else {
                &parameters
            }
        ));
        render_block(
            &mut output,
            module,
            function.body.as_ref().expect("filtered source body"),
        )?;
        if function.result == LoweredType::Unit {
            output.push_str("    return;\n");
        }
        output.push_str("}\n\n");
    }

    output.push_str("int main(int argc, char **argv) {\n    (void)argc;\n    (void)argv;\n    if (__sev_init() != 0) return 1;\n");
    if let Some(entry) = module.entry {
        let function = function(module, entry)?;
        match function.parameters.as_slice() {
            [] => output.push_str(&format!("    {}();\n", function_name(function))),
            [parameter] if value_type(module, *parameter)? == LoweredType::Arguments => {
                output.push_str("    sev_args __sev_args = { argc - 1, argv + 1 };\n");
                output.push_str(&format!("    {}(__sev_args);\n", function_name(function)));
            }
            _ => {
                return Err(BackendError::UnsupportedOperation(
                    "entry must be `main()` or `main(args: args)`".into(),
                ))
            }
        }
    }
    output.push_str("    return 0;\n}\n");
    Ok(output)
}

fn render_block(
    output: &mut String,
    module: &LoweredModule,
    block: &Block,
) -> Result<(), BackendError> {
    for operation in &block.operations {
        match operation {
            Operation::Coverage { key } => {
                output.push_str(&format!(
                    "    __sev_coverage_hit({});\n",
                    c_string_literal(key)
                ));
            }
            Operation::Constant { value, result } => {
                let ty = value_type(module, *result)?;
                let literal = c_literal(value, ty)?;
                define_value(output, module, *result, &literal)?;
            }
            Operation::Unary {
                operator,
                operand,
                result,
            } => {
                define_value(
                    output,
                    module,
                    *result,
                    &format!("{}v{}", c_unary(*operator), operand.0),
                )?;
            }
            Operation::Convert { operand, result } => {
                define_value(
                    output,
                    module,
                    *result,
                    &format!("({})v{}", c_type(value_type(module, *result)?)?, operand.0),
                )?;
            }
            Operation::Binary {
                operator,
                left,
                right,
                result,
            } => {
                let result_type = value_type(module, *result)?;
                let left_type = value_type(module, *left)?;
                let right_type = value_type(module, *right)?;
                if matches!(left_type, LoweredType::String | LoweredType::Bytes)
                    || matches!(right_type, LoweredType::String | LoweredType::Bytes)
                {
                    return Err(BackendError::UnsupportedOperation(
                        "string/byte operations require a lowered runtime interface".into(),
                    ));
                }
                let _ = result_type;
                define_value(
                    output,
                    module,
                    *result,
                    &format!("v{} {} v{}", left.0, c_binary(*operator)?, right.0),
                )?;
            }
            Operation::Aggregate { .. }
            | Operation::FieldGet { .. }
            | Operation::FieldSet { .. } => {
                return Err(BackendError::UnsupportedOperation(
                    "aggregate classes require the MLIR backend".into(),
                ));
            }
            Operation::Load { .. } | Operation::Store { .. } => {
                return Err(BackendError::UnsupportedOperation(
                    "place-based LIR requires the CFG backend".into(),
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
                    .map(|argument| format!("v{}", argument.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let call = format!("{}({arguments})", function_name(target));
                if value_type(module, *result)? == LoweredType::Unit {
                    output.push_str(&format!("    {call};\n"));
                } else {
                    define_value(output, module, *result, &call)?;
                }
            }
            Operation::Spawn { .. } | Operation::Await { .. } => {
                return Err(BackendError::UnsupportedOperation(
                    "task operations require the MLIR async lowering".into(),
                ));
            }
            Operation::RuntimeCall {
                symbol,
                arguments,
                result,
            } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| format!("v{}", argument.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let call = format!("{symbol}({arguments})");
                if let Some(result) = result {
                    define_value(output, module, *result, &call)?;
                } else {
                    output.push_str(&format!("    {call};\n"));
                }
            }
            Operation::Return { value } => match value {
                Some(value) => output.push_str(&format!("    return v{};\n", value.0)),
                None => output.push_str("    return;\n"),
            },
            Operation::Assert {
                condition,
                message,
                location,
            } => {
                output.push_str(&format!("    if (!v{}) {{\n", condition.0));
                let failure = location.as_ref().map(|location| {
                    format!(
                        "{}:{}:{}: assertion failed: {}",
                        location.file, location.line, location.column, location.expression
                    )
                });
                match (failure, message) {
                    (Some(failure), Some(message)) => output.push_str(&format!(
                        "        fprintf(stderr, \"%s: %s\\n\", {}, v{});\n",
                        c_string_literal(&failure),
                        message.0
                    )),
                    (Some(failure), None) => output.push_str(&format!(
                        "        fputs({}, stderr);\n",
                        c_string_literal(&format!("{failure}\n"))
                    )),
                    (None, Some(message)) => output.push_str(&format!(
                        "        fprintf(stderr, \"assertion failed: %s\\n\", v{});\n",
                        message.0
                    )),
                    (None, None) => {
                        output.push_str("        fputs(\"assertion failed\\n\", stderr);\n");
                    }
                }
                output.push_str("        exit(1);\n    }\n");
            }
            Operation::If {
                condition,
                then_block,
                else_block,
            } => {
                output.push_str(&format!("    if (v{}) {{\n", condition.0));
                render_block(output, module, then_block)?;
                if else_block.operations.is_empty() {
                    output.push_str("    }\n");
                } else {
                    output.push_str("    } else {\n");
                    render_block(output, module, else_block)?;
                    output.push_str("    }\n");
                }
            }
            Operation::While {
                condition_block,
                condition,
                body,
            } => {
                output.push_str("    while (1) {\n");
                render_block(output, module, condition_block)?;
                output.push_str(&format!("        if (!v{}) break;\n", condition.0));
                render_block(output, module, body)?;
                output.push_str("    }\n");
            }
            Operation::Break => output.push_str("    break;\n"),
            Operation::Continue => output.push_str("    continue;\n"),
            Operation::ArtifactCall { artifact, .. } => {
                return Err(BackendError::UnsupportedOperation(format!(
                    "artifact call `{artifact:?}` requires the MLIR composition pipeline"
                )))
            }
        }
    }
    Ok(())
}

fn define_value(
    output: &mut String,
    module: &LoweredModule,
    result: ValueId,
    expression: &str,
) -> Result<(), BackendError> {
    if module.globals.contains(&result) {
        output.push_str(&format!("    v{} = {expression};\n", result.0));
    } else {
        output.push_str(&format!(
            "    {} v{} = {expression};\n",
            c_type(value_type(module, result)?)?,
            result.0
        ));
    }
    Ok(())
}

fn function(module: &LoweredModule, id: FunctionId) -> Result<&Function, BackendError> {
    module
        .functions
        .iter()
        .find(|function| function.id == id)
        .ok_or_else(|| BackendError::UnsupportedOperation(format!("unknown LIR function {}", id.0)))
}

fn all_operations(module: &LoweredModule) -> Vec<&Operation> {
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

fn function_name(function: &Function) -> String {
    match &function.linkage {
        FunctionLinkage::Internal => format!("__sev_fn_{}", function.id.0),
        FunctionLinkage::External { symbol } => symbol.clone(),
    }
}

fn c_return_type(ty: LoweredType) -> Result<&'static str, BackendError> {
    if ty == LoweredType::Unit {
        Ok("void")
    } else {
        c_type(ty)
    }
}

pub fn supports_direct_lir(module: &LoweredModule) -> bool {
    module.initializer_cfg.is_none() && render_c(module).is_ok()
}

fn value_type(module: &LoweredModule, id: ValueId) -> Result<LoweredType, BackendError> {
    module
        .values
        .get(id.0 as usize)
        .filter(|value| value.id == id)
        .map(|value| value.ty)
        .ok_or(BackendError::InvalidValue(id))
}

fn c_type(ty: LoweredType) -> Result<&'static str, BackendError> {
    match ty {
        LoweredType::Integer {
            bits: 8,
            signed: true,
        } => Ok("int8_t"),
        LoweredType::Integer {
            bits: 16,
            signed: true,
        } => Ok("int16_t"),
        LoweredType::Integer {
            bits: 32,
            signed: true,
        } => Ok("int32_t"),
        LoweredType::Integer {
            bits: 64,
            signed: true,
        } => Ok("int64_t"),
        LoweredType::Integer {
            bits: 128,
            signed: true,
        } => Ok("__int128"),
        LoweredType::Integer {
            bits: 8,
            signed: false,
        } => Ok("uint8_t"),
        LoweredType::Integer {
            bits: 16,
            signed: false,
        } => Ok("uint16_t"),
        LoweredType::Integer {
            bits: 32,
            signed: false,
        } => Ok("uint32_t"),
        LoweredType::Integer {
            bits: 64,
            signed: false,
        } => Ok("uint64_t"),
        LoweredType::Integer {
            bits: 128,
            signed: false,
        } => Ok("unsigned __int128"),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        } => Ok("float"),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(64),
        } => Ok("double"),
        LoweredType::Boolean => Ok("_Bool"),
        LoweredType::String => Ok("const char *"),
        LoweredType::None | LoweredType::Unit => Ok("uint8_t"),
        LoweredType::Arguments => Ok("sev_args"),
        unsupported => Err(BackendError::UnsupportedType(unsupported)),
    }
}

fn c_literal(value: &Constant, ty: LoweredType) -> Result<String, BackendError> {
    match (value, ty) {
        (Constant::Integer(spelling), LoweredType::Integer { bits, signed }) => {
            c_integer_literal(spelling, bits, signed)
        }
        (Constant::Float(spelling), LoweredType::Float { .. }) => Ok(spelling.clone()),
        (Constant::Boolean(value), LoweredType::Boolean) => {
            Ok(if *value { "1" } else { "0" }.into())
        }
        (Constant::String(value), LoweredType::String) => Ok(c_string_literal(value)),
        (Constant::None, LoweredType::None) | (Constant::Unit, LoweredType::Unit) => Ok("0".into()),
        _ => Err(BackendError::UnsupportedOperation(format!(
            "C backend cannot emit {value:?} as {ty:?}"
        ))),
    }
}

fn c_integer_literal(spelling: &str, bits: u16, signed: bool) -> Result<String, BackendError> {
    if bits != 128 {
        return Ok(spelling.to_owned());
    }
    let value = if signed {
        spelling.parse::<i128>().map(|value| value as u128)
    } else {
        spelling.parse::<u128>()
    }
    .map_err(|_| {
        BackendError::UnsupportedOperation(format!(
            "invalid {}128 integer literal `{spelling}`",
            if signed { "i" } else { "u" }
        ))
    })?;
    let high = value >> 64;
    let low = value as u64;
    let limbs = format!(
        "((((unsigned __int128)UINT64_C(0x{high:016x})) << 64) | (unsigned __int128)UINT64_C(0x{low:016x}))"
    );
    Ok(if signed {
        format!("((__int128){limbs})")
    } else {
        limbs
    })
}

fn c_string_literal(value: &str) -> String {
    let mut output = String::from("\"");
    for byte in value.bytes() {
        match byte {
            b'\\' => output.push_str("\\\\"),
            b'\"' => output.push_str("\\\""),
            b'\n' => output.push_str("\\n"),
            b'\r' => output.push_str("\\r"),
            b'\t' => output.push_str("\\t"),
            0x20..=0x7e => output.push(char::from(byte)),
            _ => output.push_str(&format!("\\{:03o}", byte)),
        }
    }
    output.push('\"');
    output
}

fn c_unary(operator: UnaryOperation) -> &'static str {
    match operator {
        UnaryOperation::Positive => "+",
        UnaryOperation::Negative => "-",
        UnaryOperation::Not => "!",
    }
}

fn c_binary(operator: BinaryOperation) -> Result<&'static str, BackendError> {
    Ok(match operator {
        BinaryOperation::Add => "+",
        BinaryOperation::Subtract => "-",
        BinaryOperation::Multiply => "*",
        BinaryOperation::Divide => "/",
        BinaryOperation::Remainder => "%",
        BinaryOperation::Power => {
            return Err(BackendError::UnsupportedOperation(
                "power requires a lowered runtime or target operation".into(),
            ))
        }
        BinaryOperation::Equal => "==",
        BinaryOperation::NotEqual => "!=",
        BinaryOperation::Less => "<",
        BinaryOperation::LessEqual => "<=",
        BinaryOperation::Greater => ">",
        BinaryOperation::GreaterEqual => ">=",
        BinaryOperation::Contains => {
            return Err(BackendError::UnsupportedOperation(
                "containment requires a lowered collection or test-runner operation".into(),
            ))
        }
        BinaryOperation::And => "&&",
        BinaryOperation::Or => "||",
    })
}

pub fn emit_executable(module: &LoweredModule, output: &Path) -> Result<Artifact, BackendError> {
    let source = render_c(module)?;
    let mut child = Command::new("cc")
        .args(["-std=c11", "-x", "c", "-"])
        .args(severian_runtime::native_sources())
        .arg("-lm")
        .arg("-o")
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(BackendError::Spawn)?;
    child
        .stdin
        .take()
        .expect("piped stdin is available")
        .write_all(source.as_bytes())
        .map_err(BackendError::Write)?;
    let result = child.wait_with_output().map_err(BackendError::Wait)?;
    if !result.status.success() {
        return Err(BackendError::CompilerFailed(
            String::from_utf8_lossy(&result.stderr).trim().to_owned(),
        ));
    }
    Ok(Artifact {
        path: output.to_owned(),
        kind: ArtifactKind::Executable,
    })
}

/// Lowers a verified, composed MLIR module through the host MLIR/LLVM
/// toolchain. This boundary accepts physical IR and never interprets Severian
/// types or CompileType routes.
pub fn emit_mlir_executable(
    module: &str,
    target_triple: &str,
    output: &Path,
) -> Result<Artifact, BackendError> {
    let lowered = run_tool(
        "mlir-opt",
        tool("SEVERIAN_MLIR_OPT", "mlir-opt-21"),
        &[
            "--verify-each",
            "--async-to-async-runtime",
            "--async-runtime-ref-counting",
            "--async-runtime-ref-counting-opt",
            "--convert-async-to-llvm",
            "--convert-scf-to-cf",
            "--convert-math-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        ],
        module.as_bytes(),
    )?;
    let llvm_ir = run_tool(
        "mlir-translate",
        tool("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21"),
        &["--mlir-to-llvmir"],
        &lowered,
    )?;
    let target = format!("--target={target_triple}");
    let output_path = output.to_string_lossy().into_owned();
    let mut clang_arguments = vec![
        target,
        "-pthread".into(),
        "-x".into(),
        "ir".into(),
        "-".into(),
        "-x".into(),
        "c".into(),
    ];
    clang_arguments.extend(
        severian_runtime::native_sources()
            .into_iter()
            .map(|source| source.to_string_lossy().into_owned()),
    );
    if module.contains("async.") {
        let llvm_config = tool("SEVERIAN_LLVM_CONFIG", "llvm-config-21");
        let libdir_output = Command::new(&llvm_config)
            .arg("--libdir")
            .output()
            .map_err(|source| BackendError::ToolSpawn {
                tool: "llvm-config",
                source,
            })?;
        if !libdir_output.status.success() {
            return Err(BackendError::ToolFailed {
                tool: "llvm-config",
                diagnostic: String::from_utf8_lossy(&libdir_output.stderr)
                    .trim()
                    .to_owned(),
            });
        }
        let libdir = String::from_utf8_lossy(&libdir_output.stdout)
            .trim()
            .to_owned();
        clang_arguments.extend([
            format!("-L{libdir}"),
            format!("-Wl,-rpath,{libdir}"),
            "-lmlir_async_runtime".into(),
        ]);
    }
    clang_arguments.extend(["-lm".into(), "-o".into(), output_path]);
    let clang_arguments = clang_arguments
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    run_tool(
        "clang",
        tool("SEVERIAN_CLANG", "clang-21"),
        &clang_arguments,
        &llvm_ir,
    )?;
    Ok(Artifact {
        path: output.to_owned(),
        kind: ArtifactKind::Executable,
    })
}

fn tool(variable: &str, default: &'static str) -> String {
    std::env::var(variable).unwrap_or_else(|_| default.to_owned())
}

fn run_tool(
    name: &'static str,
    program: String,
    arguments: &[&str],
    input: &[u8],
) -> Result<Vec<u8>, BackendError> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| BackendError::ToolSpawn { tool: name, source })?;
    child
        .stdin
        .take()
        .expect("piped stdin is available")
        .write_all(input)
        .map_err(|source| BackendError::ToolWrite { tool: name, source })?;
    let result = child
        .wait_with_output()
        .map_err(|source| BackendError::ToolWait { tool: name, source })?;
    if !result.status.success() {
        return Err(BackendError::ToolFailed {
            tool: name,
            diagnostic: String::from_utf8_lossy(&result.stderr).trim().to_owned(),
        });
    }
    Ok(result.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_lir::Value;

    #[test]
    fn c_spelling_is_selected_in_the_c_emitter() {
        let module = LoweredModule {
            values: vec![Value {
                id: ValueId(0),
                ty: LoweredType::Integer {
                    bits: 32,
                    signed: true,
                },
            }],
            globals: vec![],
            initializer: Block {
                operations: vec![Operation::Constant {
                    value: Constant::Integer("10".into()),
                    result: ValueId(0),
                }],
            },
            functions: vec![],
            entry: None,
            traits: vec![],
            classes: vec![],
            storage_globals: vec![],
            initializer_cfg: None,
        };
        assert!(render_c(&module).unwrap().contains("int32_t v0 = 10;"));
    }

    #[test]
    fn unsupported_widths_and_bfloat_are_errors() {
        assert!(matches!(
            c_type(LoweredType::Integer {
                bits: 24,
                signed: true
            }),
            Err(BackendError::UnsupportedType(_))
        ));
        assert!(matches!(
            c_type(LoweredType::Float {
                format: LoweredFloatFormat::BrainFloat16
            }),
            Err(BackendError::UnsupportedType(_))
        ));
    }

    #[test]
    fn c_strings_are_escaped_without_changing_their_bytes() {
        assert_eq!(c_string_literal("a\n\"b\\c"), "\"a\\n\\\"b\\\\c\"");
        assert_eq!(c_string_literal("é"), "\"\\303\\251\"");
    }

    #[test]
    fn c_emitter_handles_lowered_runtime_calls_generically() {
        let module = LoweredModule {
            values: (0..3)
                .map(|id| Value {
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
            ..LoweredModule::default()
        };
        let rendered = render_c(&module).unwrap();
        assert!(rendered.contains("__sev_string_concat(v0, v1)"));
        assert!(rendered.contains("const char * __sev_string_concat(const char *, const char *);"));
    }

    #[test]
    fn i128_literals_are_composed_from_exact_u64_limbs() {
        assert_eq!(
            c_integer_literal(&i128::MIN.to_string(), 128, true).unwrap(),
            "((__int128)((((unsigned __int128)UINT64_C(0x8000000000000000)) << 64) | (unsigned __int128)UINT64_C(0x0000000000000000)))"
        );
        assert_eq!(
            c_integer_literal(&i128::MAX.to_string(), 128, true).unwrap(),
            "((__int128)((((unsigned __int128)UINT64_C(0x7fffffffffffffff)) << 64) | (unsigned __int128)UINT64_C(0xffffffffffffffff)))"
        );
        assert_eq!(
            c_integer_literal(&u128::MAX.to_string(), 128, false).unwrap(),
            "((((unsigned __int128)UINT64_C(0xffffffffffffffff)) << 64) | (unsigned __int128)UINT64_C(0xffffffffffffffff))"
        );
    }
}
