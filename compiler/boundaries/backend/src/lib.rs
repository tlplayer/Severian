#![forbid(unsafe_code)]

use severian_lir::LoweredFloatFormat;
pub use severian_lir::{
    BinaryOperation, Constant, LoweredType, Module as LoweredModule, Operation, UnaryOperation,
    ValueId,
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
        }
    }
}

impl std::error::Error for BackendError {}

pub fn render_c(module: &LoweredModule) -> Result<String, BackendError> {
    let mut output = String::from("#include <stdint.h>\n\nint main(void) {\n");
    for operation in &module.operations {
        match operation {
            Operation::Constant { value, result } => {
                let ty = value_type(module, *result)?;
                let spelling = c_type(ty)?;
                let literal = c_literal(value, ty)?;
                output.push_str(&format!("    {spelling} v{} = {literal};\n", result.0));
            }
            Operation::Unary {
                operator,
                operand,
                result,
            } => {
                let spelling = c_type(value_type(module, *result)?)?;
                output.push_str(&format!(
                    "    {spelling} v{} = {}v{};\n",
                    result.0,
                    c_unary(*operator),
                    operand.0
                ));
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
                let spelling = c_type(result_type)?;
                output.push_str(&format!(
                    "    {spelling} v{} = v{} {} v{};\n",
                    result.0,
                    left.0,
                    c_binary(*operator)?,
                    right.0
                ));
            }
        }
    }
    if let Some(value) = module.last_binding {
        output.push_str(&format!("    (void)v{};\n", value.0));
    }
    output.push_str("    return 0;\n}\n");
    Ok(output)
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
        _ => Err(BackendError::UnsupportedOperation(format!(
            "C backend cannot emit {value:?} as {ty:?}"
        ))),
    }
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
            operations: vec![Operation::Constant {
                value: Constant::Integer("10".into()),
                result: ValueId(0),
            }],
            last_binding: Some(ValueId(0)),
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
}
