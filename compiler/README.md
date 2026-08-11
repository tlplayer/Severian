# Severian Compiler

## Architecture

```text
Source
  ↓
Lexer
  ↓
Parser / AST
  ↓
Resolve / typecheck (currently one semantic stage)
  ↓
Typed HIR
  ↓
Ownership analysis
  ↓
HIR optimization passes
  ↓
MIR control-flow graph
  ↓
Lowering
  ├── Native MLIR → LLVM → native backend → executable
  └── StableHLO → XLA → PJRT → device executable and buffers
```

The workspace keeps language concerns separated:

- `lexer`, `parser`, and `ast` implement source syntax.
- `source` owns file identities, byte spans, and line/column mapping shared by
  diagnostics and lowering locations.
- `package` resolves manifests, dependency source, public interfaces, symbol
  packs, and compiler contracts declared by libraries.
- `semantic` resolves those interfaces into typed HIR; `ownership` checks HIR.
- `passes` currently owns the working dataflow and loop transformations over
  HIR. Domain packages register compatible operations through manifest metadata
  rather than hard-coded driver function names.
- `mir` owns the explicit function/basic-block control-flow graph and typed HIR
  value references. It is a mandatory, independently emittable native-pipeline
  stage.
- `lowering` accepts MIR, converts its structured expression payload to MLIR,
  and generates compiler-owned ABI glue.
- `xla` owns StableHLO artifacts and the PJRT compilation/runtime boundary.
  Optimization after StableHLO handoff belongs to XLA itself.
- `platform` contains concrete native providers such as SQLite and ranked
  tensor/memref bridges.
- `backend` verifies MLIR, translates it to LLVM IR, and invokes the native
  linker.
- `runtime` owns native runtime symbols and behavior only. Compiler-side
  runtime-call emission lives under `lowering::runtime`.
- `driver` composes the stages and provides the CLI. It has no interpreter or
  evaluator; execution always means launching a compiled artifact.

The native path remains the default. The XLA path is selected independently
for tensor/ML workloads; StableHLO emission is wired for directly returned,
typed tensor calls. General tensor bodies still require expression result types
to be retained in HIR. PJRT execution is not exposed yet; plugin selection in
the driver will require an explicit path or Severian environment variable and
will not scan the filesystem implicitly.

HIR v2 owns semantic identity and type information: expressions carry `HirId`
and their resolved `ValueType`; functions, type definitions, and variants carry
stable IDs; resolved call targets retain their full function signature and
native ABI symbol where one exists. Lowering must consume that information
directly instead of rebuilding types or package identity from names. MIR now
owns explicit control-flow blocks while reusing typed HIR value identities;
MIR-native optimization passes and IREE support remain future work rather than
empty active compiler modules.

Direct GPU lowering and ROCm backend implementation remain as inactive
low-level code because they contain real MLIR/ROCDL lowering work. They are not
driver targets in the current Native/XLA architecture.

`sev check` stops after semantic and ownership checks. `sev build`, `sev run`,
and `sev test` continue through lowering and the native backend; `run` and
`test` use the resulting OS executable as the only source of execution truth.

## Local commands

```bash
cargo run -p severian-driver --bin sev -- check docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- run docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- test docs/examples/01-values-control/03-basic-functions.sev
cargo run -p severian-driver --bin sev -- coverage docs/examples/26-problems
cargo run -p severian-driver --bin sev -- test docs/examples/26-problems/06-min-cost-climbing-stairs.sev --mutate
cargo run -p severian-driver --bin sev -- memory docs/examples/08-concurrency/10-locked-shared-mutation.sev --sanitizer thread
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
