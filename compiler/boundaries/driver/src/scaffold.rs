pub(super) fn package_manifest(name: &str) -> String {
    format!(
        r#"# Severian package manifest.
# `sev init` writes the complete project contract so available controls are
# discoverable here. Tighten the lenient starting values as the project matures.

[package]
name = "{name}"
version = "0.1.0"
edition = "2026"
# Optional package metadata used by registries and documentation.
# description = ""
# license = "Severian License"
# repository = "https://example.com/owner/project"

[[bin]]
name = "{name}"
path = "src/main.sev"

[compiler.type_resolution]
# Explicit `Any` remains legal. These switches reject dynamic types created by
# failed or incomplete compiler resolution.
deny_any = false
deny_tensor_any = false
deny_unresolved = false
deny_inferred_fallback = false
deny_lost_type_information = false

# For a library target, create src/lib.sev and enable this table.
# [lib]
# path = "src/lib.sev"

[build]
# `user` shows source-level diagnostics. `internal` also includes compiler data
# useful when reporting or debugging a compiler/backend failure. The CLI
# override is `sev build --diagnostics user|internal`.
diagnostics = "user"
# Build command defaults; explicit CLI flags override these values.
emit = "executable" # executable, hir, mir, mlir, stablehlo, llvm, or asm
target = "native" # native/cpu or xla (including xla:<device>)
max_errors = 50
message_format = "text" # text or json
verify_each = false
# Artifact placement and parallelism are available to build-system clients.
target_directory = "target"
# jobs = 4
# `sev build` implicitly applies formatting and lint fixes before the mandatory
# verification gates. Declare `pipeline` only to disable a mutation stage or to
# override the default policy; all verification gates remain mandatory.
# pipeline = [
#     "format", "lint", "compile", "architecture", "test", "profile",
#     "coverage", "memory", "integ",
# ]

# New applications begin permissively. Raise these percentages as tests land.
[coverage]
minimum = 0
changed_minimum = 0
regions = 0
branches = 0
functions = 0
per_file = true

[memory]
# Change to `deny` once the project is ready to make leaks build-blocking.
leaks = "allow"

[architecture]
# Dependency cycles are rejected for local Cargo and Severian packages. Layer
# enforcement becomes active when an order is declared below.
enforce = true
deny_cycles = true
deny_unknown_layers = false
deny_layer_violations = true

# [architecture.layers]
# include = ["compiler/*"]
# order = ["syntax", "semantic", "ir", "backend"]

# Explicit allow lists and denials are evaluated against resolved package edges.
# [[architecture.rule]]
# from = "compiler/backend/**"
# allow = ["compiler/ir/**"]
# deny = ["compiler/syntax/**"]

[architecture.files]
# These broad limits make growth visible without constraining early exploration.
# Lower them toward 500/800 as module boundaries stabilize.
soft_lines = 2000
hard_lines = 4000
include = ["src/**/*.sev", "tests/**/*.sev"]

# Time-bounded exceptions can temporarily replace the limits above.
# [[architecture.files.exception]]
# path = "src/legacy.sev"
# soft_lines = 2500
# hard_lines = 5000
# reason = "split into focused modules before the next release"
# owner = "team-name"
# expires = "2027-01-01"

# Profiles expose every supported code-generation control. A sanitizer may be
# `address`, `thread`, `memory`, or `undefined`; leave it unset for normal builds.
[profile.dev]
optimization = 0
debug = "full"
lto = "off"
incremental = true
overflow_checks = true
assertions = true
runtime_checks = true
coverage = false
# sanitizer = "address"

[profile.release]
optimization = 3
debug = "line-tables"
lto = "thin"
incremental = false
overflow_checks = false
assertions = false
runtime_checks = false
coverage = false
# sanitizer = "address"

[profile.test]
inherits = "dev"

[profile.bench]
inherits = "release"

[profile.coverage]
inherits = "dev"
incremental = false
coverage = true

[dependencies]
# registry-package = "1.2.3"
# local-package = {{ path = "../local-package", version = "0.1.0" }}
# git-package = {{ git = "https://example.com/owner/repo", branch = "main" }}
# Optional dependency fields: package, registry, rev, tag, optional, features,
# and default_features.

[features]
default = []
# extra = ["optional-dependency"]

[lints]
# Project lint levels belong here as lint groups become available.

# A workspace-only manifest can expose these fields instead of package targets.
# [workspace]
# members = ["packages/*"]
# exclude = []
# default_members = []

# Unsafe capabilities stay denied unless an exact source exception is declared.
# Supported capabilities: native-abi, pointers, runtime-owned-tasks,
# and unsafe-blocks.
# [package.unsafe]
# capabilities = ["pointers"]
# sources = ["src/platform.sev"]

# External installers are opt-in and normally added by `sev add`/`sev install`.
# [system]
# example-tool = ">=1.0"
# [install.example-tool]
# publisher = "verified-publisher"
# package = "vendor-package-name"
# source = "vendor"
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_manifest_is_valid_and_starts_lenient() {
        let source = package_manifest("example-app");
        let manifest = toml::from_str::<toml::Value>(&source).unwrap();
        assert_eq!(
            manifest["compiler"]["type_resolution"]["deny_any"].as_bool(),
            Some(false)
        );
        assert_eq!(manifest["build"]["diagnostics"].as_str(), Some("user"));
        assert_eq!(manifest["coverage"]["minimum"].as_integer(), Some(0));
        assert_eq!(manifest["coverage"]["regions"].as_integer(), Some(0));
        assert_eq!(manifest["coverage"]["branches"].as_integer(), Some(0));
        assert_eq!(manifest["coverage"]["functions"].as_integer(), Some(0));
        assert_eq!(manifest["coverage"]["per_file"].as_bool(), Some(true));
        assert_eq!(manifest["memory"]["leaks"].as_str(), Some("allow"));
        assert_eq!(manifest["architecture"]["enforce"].as_bool(), Some(true));
        assert_eq!(
            manifest["architecture"]["deny_cycles"].as_bool(),
            Some(true)
        );
        assert_eq!(manifest["build"]["max_errors"].as_integer(), Some(50));
        assert_eq!(manifest["features"]["default"].as_array().unwrap().len(), 0);
        assert!(source.contains("# [package.unsafe]"));
        assert!(source.contains("[profile.release]"));
    }
}
