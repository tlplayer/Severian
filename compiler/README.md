# Severian Compiler

## Architecture

The workspace keeps language concerns separated:

- `lexer`, `parser`, and `ast` implement source syntax.
- `package` resolves manifests, dependency source, public interfaces, symbol
  packs, and compiler contracts declared by libraries.
- `semantic` resolves those interfaces into HIR; `ownership` checks HIR.
- `passes` owns the generic optimization pipeline. Domain packages register
  compatible operations through manifest metadata rather than hard-coded
  driver function names.
- `lowering` converts HIR to MLIR and generates compiler-owned ABI glue.
- `platform` contains concrete native providers such as SQLite and ranked
  tensor/memref bridges.
- `backend` verifies MLIR, translates it to LLVM IR, and invokes the native
  linker.
- `driver` composes the stages and provides the CLI and controlled evaluator.

The controlled evaluator runs the checked, unoptimized HIR. Native compilation
runs an optimized clone through lowering, so backend-only operations do not
have a second model-specific implementation in the evaluator.

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
