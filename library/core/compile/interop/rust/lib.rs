#![forbid(unsafe_code)]

pub const PACKAGE_NAME: &str = "core.compile";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileProtocolSource {
    pub path: &'static str,
    pub source: &'static str,
}

/// Authoritative Severian protocol sources. This package does not interpret
/// declarations or assign semantic identities.
pub const SOURCES: &[CompileProtocolSource] = &[CompileProtocolSource {
    path: "src/mod.sev",
    source: include_str!("../../src/mod.sev"),
}];
