use super::{StableHloFormat, StableHloModule, StableHloVersion};
use crate::{Result, XlaError};
use std::{
    io::Write,
    process::{Command, Stdio},
};

#[derive(Debug, Clone)]
pub struct PortableArtifactOptions {
    pub stablehlo_translate: String,
    pub target_version: Option<String>,
}

impl Default for PortableArtifactOptions {
    fn default() -> Self {
        Self {
            stablehlo_translate: "stablehlo-translate".into(),
            target_version: None,
        }
    }
}

/// Serializes StableHLO to the versioned portable artifact format.
///
/// StableHLO's public compatibility tooling exposes this through
/// `stablehlo-translate --serialize`. Keeping this subprocess boundary here
/// lets Severian use the official serializer now and replace it with direct
/// C++/C bindings later without changing the compiler/runtime API.
pub fn serialize_portable(
    module: &StableHloModule,
    options: &PortableArtifactOptions,
) -> Result<StableHloModule> {
    if module.format() == StableHloFormat::PortableArtifact {
        return Ok(module.clone());
    }

    let textual = match module.format() {
        StableHloFormat::Text => module.clone(),
        StableHloFormat::MlirBytecode => super::import::deserialize_to_text(module)?,
        StableHloFormat::PortableArtifact => unreachable!(),
    };

    let mut command = Command::new(&options.stablehlo_translate);
    command.arg("--serialize");

    let module_target_version = module.target_version().map(ToString::to_string);
    if let Some(version) = options
        .target_version
        .as_deref()
        .or(module_target_version.as_deref())
    {
        command.arg(format!("--target={version}"));
    }

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|error| {
        XlaError::StableHloTool(format!(
            "failed to start {}: {error}",
            options.stablehlo_translate
        ))
    })?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| XlaError::StableHloTool("serializer stdin unavailable".into()))?
        .write_all(textual.bytes())?;

    let output = child.wait_with_output()?;

    if !output.status.success() {
        return Err(XlaError::StableHloTool(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }

    let mut artifact =
        StableHloModule::from_bytes(output.stdout, StableHloFormat::PortableArtifact);

    if let Some(version) = options.target_version.as_deref() {
        artifact = artifact.with_target_version(StableHloVersion::parse(version)?);
    }

    Ok(artifact)
}

pub fn write_module(module: &StableHloModule, path: impl AsRef<std::path::Path>) -> Result<()> {
    std::fs::write(path, module.bytes())?;
    Ok(())
}
