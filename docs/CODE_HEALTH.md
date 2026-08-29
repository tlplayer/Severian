# Code health

The repository's code-health policy is implemented by
[`tools/health`](../tools/health/README.md). The command is a small analysis
compiler, not a maintainability score and not a broad Clippy preset.

Run `cargo xtask health` before review. For a pull request, use
`cargo xtask health --changed origin/main`; after producing branch-aware LLVM
coverage JSON, pass it with `--coverage`. Existing hard debt is visible but
ratcheted by stable, expiring fingerprints.

Compiler transformations are checked at their owning IR boundary. The current
MIR pass contract and the remaining stage-verifier map are documented in
[`compiler/tools/health`](../compiler/tools/health/README.md).
