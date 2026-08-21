# Bootstrapped Severian compiler

`sev_compiler` is the source-language twin of the Rust compiler in
`compiler`. File-for-file, directory-for-directory, and package-for-package
parity is the architectural invariant.

The Rust tree is the stage-0 executable specification. Each matching `.sev`
file is filled in until it has behavioral parity, after which the Severian
implementation may be simplified or improved without losing the correspondence
needed for differential testing.

## Invariants

1. Every directory under `compiler` exists at the same relative path here.
2. Every Rust `.rs` source has a `.sev` source at the same relative path.
3. Every compiler `Cargo.toml` has a `package.toml` counterpart.
4. New Rust compiler structure cannot land without updating this tree.
5. Each pair consumes equivalent fixtures and produces normalized equivalent
   results before the Rust implementation can be retired.
6. Canonical `docs/examples` sources are never copied, rewritten, or skipped.

## Bootstrap stages

- `sev0`: the Rust compiler builds the mirrored Severian package graph.
- `sev1`: the compiler produced by `sev0`.
- `sev2`: the compiler produced by `sev1`.
- `sev3`: the compiler produced by `sev2`.
- Stage 2 and stage 3 establish the self-hosted fixed point.

Run the structural and stage-0 gates with:

```bash
python3 tools/quality/check_bootstrap_mirror.py
cargo run -q -p severian-driver -- check sev_compiler/boundaries/driver
cargo run -q -p severian-driver -- build sev_compiler/boundaries/driver --bin sev-bootstrap-driver
cargo run -q -p severian-driver -- test test/validation/examples
```

The canonical suite currently exposes the language work still required. That
red result is the bootstrap backlog; it must not be converted into skips.
