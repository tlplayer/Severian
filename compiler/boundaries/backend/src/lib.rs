#![forbid(unsafe_code)]

#[path = "model/artifact/mod.rs"]
mod artifact;
#[path = "model/capability/mod.rs"]
mod capability;

pub use artifact::{Artifact, ArtifactKind};
pub use capability::{LoweredModule, LoweredType, Operation, ValueId};
use severian_abi::NativeType;
use std::fmt;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug)]
pub enum BackendError {
    Spawn(std::io::Error),
    Write(std::io::Error),
    Wait(std::io::Error),
    CompilerFailed(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(error) => write!(formatter, "could not start native C compiler: {error}"),
            Self::Write(error) => write!(
                formatter,
                "could not send lowered program to native C compiler: {error}"
            ),
            Self::Wait(error) => write!(formatter, "could not wait for native C compiler: {error}"),
            Self::CompilerFailed(error) => write!(formatter, "native C compiler failed: {error}"),
        }
    }
}

pub fn render_c(module: &LoweredModule) -> String {
    let integer = NativeType::I64.c_spelling();
    let mut output = String::from("#include <stdint.h>\n\nint main(void) {\n");
    for operation in &module.operations {
        match operation {
            Operation::ConstantI64 { value, result } => {
                output.push_str(&format!("    {integer} v{} = {value};\n", result.0))
            }
            Operation::AddI64 {
                left,
                right,
                result,
            } => output.push_str(&format!(
                "    {integer} v{} = v{} + v{};\n",
                result.0, left.0, right.0
            )),
        }
    }
    if let Some(value) = module.last_binding {
        output.push_str(&format!("    (void)v{};\n", value.0));
    }
    output.push_str("    return 0;\n}\n");
    output
}

pub fn emit_executable(module: &LoweredModule, output: &Path) -> Result<Artifact, BackendError> {
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
        .write_all(render_c(module).as_bytes())
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
    #[test]
    fn c_output_contains_concrete_integer_addition() {
        let module = LoweredModule {
            values: vec![LoweredType::I64; 3],
            operations: vec![
                Operation::ConstantI64 {
                    value: 2,
                    result: ValueId(0),
                },
                Operation::ConstantI64 {
                    value: 1,
                    result: ValueId(1),
                },
                Operation::AddI64 {
                    left: ValueId(1),
                    right: ValueId(0),
                    result: ValueId(2),
                },
            ],
            last_binding: Some(ValueId(2)),
        };
        assert!(render_c(&module).contains("int64_t v2 = v1 + v0;"));
    }
}
