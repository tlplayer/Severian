use super::{StableHloFormat, StableHloModule};
use crate::{Result, XlaError};
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

pub fn read_module(path: impl AsRef<Path>) -> Result<StableHloModule> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let format = detect_format(&bytes, path);

    Ok(StableHloModule::from_bytes(bytes, format).with_source_name(path))
}

pub fn detect_format(bytes: &[u8], path: &Path) -> StableHloFormat {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "mlir")
        && !bytes.starts_with(b"ML\xefR")
    {
        return StableHloFormat::Text;
    }

    if bytes.starts_with(b"ML\xefR") {
        // StableHLO portable artifacts are also MLIR bytecode. Use file naming
        // as a hint; callers may override by constructing StableHloModule.
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("portable") || name.ends_with(".mlir.bc"))
        {
            StableHloFormat::PortableArtifact
        } else {
            StableHloFormat::MlirBytecode
        }
    } else {
        StableHloFormat::Text
    }
}

pub fn deserialize_to_text(module: &StableHloModule) -> Result<StableHloModule> {
    match module.format() {
        StableHloFormat::Text => return Ok(module.clone()),
        StableHloFormat::MlirBytecode | StableHloFormat::PortableArtifact => {}
    }

    run_deserializer(module, "stablehlo-translate")
}

pub fn deserialize_with_tool(
    module: &StableHloModule,
    stablehlo_translate: &str,
) -> Result<StableHloModule> {
    run_deserializer(module, stablehlo_translate)
}

fn run_deserializer(
    module: &StableHloModule,
    stablehlo_translate: &str,
) -> Result<StableHloModule> {
    let mut child = Command::new(stablehlo_translate)
        .arg("--deserialize")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            XlaError::StableHloTool(format!("failed to start {stablehlo_translate}: {error}"))
        })?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| XlaError::StableHloTool("deserializer stdin unavailable".into()))?
        .write_all(module.bytes())?;

    let output = child.wait_with_output()?;

    if !output.status.success() {
        return Err(XlaError::StableHloTool(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }

    let text = String::from_utf8(output.stdout)
        .map_err(|error| XlaError::InvalidStableHlo(error.to_string()))?;

    Ok(StableHloModule::from_text(text))
}
