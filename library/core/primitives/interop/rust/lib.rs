#![forbid(unsafe_code)]

pub const PACKAGE_NAME: &str = "core.primitives";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveSource {
    pub path: &'static str,
    pub source: &'static str,
}

/// Authoritative Severian sources. This wrapper intentionally performs no
/// parsing, ID allocation, metadata interpretation, or semantic resolution.
pub const SOURCES: &[PrimitiveSource] = &[
    PrimitiveSource {
        path: "src/mod.sev",
        source: include_str!("../../src/mod.sev"),
    },
    PrimitiveSource {
        path: "src/bool.sev",
        source: include_str!("../../src/bool.sev"),
    },
    PrimitiveSource {
        path: "src/bytes.sev",
        source: include_str!("../../src/bytes.sev"),
    },
    PrimitiveSource {
        path: "src/char.sev",
        source: include_str!("../../src/char.sev"),
    },
    PrimitiveSource {
        path: "src/floats.sev",
        source: include_str!("../../src/floats.sev"),
    },
    PrimitiveSource {
        path: "src/error.sev",
        source: include_str!("../../src/error.sev"),
    },
    PrimitiveSource {
        path: "src/integers.sev",
        source: include_str!("../../src/integers.sev"),
    },
    PrimitiveSource {
        path: "src/none.sev",
        source: include_str!("../../src/none.sev"),
    },
    PrimitiveSource {
        path: "src/measured.sev",
        source: include_str!("../../src/measured.sev"),
    },
    PrimitiveSource {
        path: "src/string.sev",
        source: include_str!("../../src/string.sev"),
    },
    PrimitiveSource {
        path: "src/unit.sev",
        source: include_str!("../../src/unit.sev"),
    },
    PrimitiveSource {
        path: "src/args.sev",
        source: include_str!("../../src/args.sev"),
    },
];
