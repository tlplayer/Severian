use crate::node::{BuildNode, BuildStage};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    fs,
    io,
    path::Path,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint(pub [u8; 32]);

impl Fingerprint {
    pub fn hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    pub fn from_hex(value: &str) -> Option<Self> {
        if value.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for index in 0..32 {
            bytes[index] = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
        }
        Some(Self(bytes))
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.hex())
    }
}

#[derive(Default)]
pub struct FingerprintBuilder {
    hasher: Sha256,
}

impl FingerprintBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(mut self, label: &str, value: &str) -> Self {
        self.hasher.update(label.as_bytes());
        self.hasher.update([0]);
        self.hasher.update(value.as_bytes());
        self.hasher.update([0xff]);
        self
    }

    pub fn bytes(mut self, label: &str, value: &[u8]) -> Self {
        self.hasher.update(label.as_bytes());
        self.hasher.update([0]);
        self.hasher.update((value.len() as u64).to_le_bytes());
        self.hasher.update(value);
        self
    }

    pub fn file(self, path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Ok(self
            .text("file-path", &normalize_path(path))
            .bytes("file-bytes", &bytes))
    }

    pub fn environment(mut self, name: &str) -> Self {
        let value = std::env::var_os(name)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default();
        self = self.text("environment-name", name);
        self.text("environment-value", &value)
    }

    pub fn dependency(mut self, fingerprint: Fingerprint) -> Self {
        self.hasher.update(b"dependency\0");
        self.hasher.update(fingerprint.0);
        self
    }

    pub fn finish(self) -> Fingerprint {
        let bytes: [u8; 32] = self.hasher.finalize().into();
        Fingerprint(bytes)
    }
}

pub fn fingerprint_node(
    node: &BuildNode,
    compiler_version: &str,
    target: &str,
    profile: &str,
    flags: &[String],
    dependency_fingerprints: &[Fingerprint],
) -> io::Result<Fingerprint> {
    let mut builder = FingerprintBuilder::new()
        .text("schema", "severian-build-fingerprint-v1")
        .text("compiler", compiler_version)
        .text("package", &node.package)
        .text("target-name", &node.target)
        .text("target", target)
        .text("profile", profile)
        .text("stage", stage_name(node.stage));

    let mut sources = node.source_files.clone();
    sources.sort();
    for source in &sources {
        builder = builder.file(source)?;
    }

    let mut flags = flags.to_vec();
    flags.sort();
    for flag in flags {
        builder = builder.text("flag", &flag);
    }

    let mut dependencies = dependency_fingerprints.to_vec();
    dependencies.sort_by_key(|fingerprint| fingerprint.0);
    for dependency in dependencies {
        builder = builder.dependency(dependency);
    }

    Ok(builder.finish())
}

fn stage_name(stage: BuildStage) -> &'static str {
    match stage {
        BuildStage::Parse => "parse",
        BuildStage::Semantic => "semantic",
        BuildStage::Ownership => "ownership",
        BuildStage::Check => "check",
        BuildStage::Optimize => "optimize",
        BuildStage::Lower => "lower",
        BuildStage::Codegen => "codegen",
        BuildStage::Link => "link",
        BuildStage::Test => "test",
        BuildStage::Bench => "bench",
        BuildStage::Coverage => "coverage",
        BuildStage::Custom => "custom",
    }
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}
