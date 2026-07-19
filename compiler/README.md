# Severian Compiler

## Local commands

```bash
cargo run -p severian-driver --bin sev -- check docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- run docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- test docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- compile docs/examples/01-values-control/03-basic-functions.sev -o /tmp/severian-basic
```

`test` executes the `test:` blocks attached to declarations. A failed Severian
`assert` makes the command fail, so examples can serve as language regression
tests while the compiler grows.

## Test

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```
