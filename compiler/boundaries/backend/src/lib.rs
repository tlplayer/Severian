#![forbid(unsafe_code)]

use severian_lir::LoweredFloatFormat;
pub use severian_lir::{
    BinaryOperation, Block, Constant, Function, FunctionId, FunctionLinkage, LoweredType,
    Module as LoweredModule, Operation, UnaryOperation, ValueId,
};
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
        "#include <stdint.h>\n\ntypedef struct { int count; char **values; } sev_args;\n\n",
    );
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
        } else {
            return Err(BackendError::UnsupportedOperation(format!(
                "function `{}` requires explicit return lowering",
                function.name
            )));
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
            Operation::Binary {
                operator,
                left,
                right,
                result,
            } => {
                let result_type = value_type(module, *result)?;
                if matches!(
                    value_type(module, *left)?,
                    LoweredType::String | LoweredType::Bytes
                ) || matches!(
                    value_type(module, *right)?,
                    LoweredType::String | LoweredType::Bytes
                ) {
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
    render_c(module).is_ok()
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
        (Constant::Integer(spelling), LoweredType::Integer { .. })
        | (Constant::Float(spelling), LoweredType::Float { .. }) => Ok(spelling.clone()),
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
        BinaryOperation::And => "&&",
        BinaryOperation::Or => "||",
    })
}

pub fn emit_executable(module: &LoweredModule, output: &Path) -> Result<Artifact, BackendError> {
    let source = render_c(module)?;
    let mut child = Command::new("cc")
        .args(["-std=c11", "-x", "c", "-", "-o"])
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
    run_tool(
        "clang",
        tool("SEVERIAN_CLANG", "clang-21"),
        &[&target, "-x", "ir", "-", "-o", &output_path],
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
}
