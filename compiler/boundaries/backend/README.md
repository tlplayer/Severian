# Backend boundary

The backend boundary turns physical compiler IR into artifacts. It can emit a
supported LIR module directly or invoke the target toolchain for an already
verified and composed MLIR module.

## Owns

- Backend selection and capability reporting.
- Artifact metadata.
- External compiler/linker/tool invocation.
- Backend-specific errors.

## Does not own

- `LirType`, LIR operations, or universal types.
- Primitive definitions or catalogs.
- Literal and operator resolution.
- MLIR spelling on shared types.
- CompileType routing, artifact verification, or MLIR composition.
- C runtime semantics embedded as an expanding string inside the emitter.

Concrete emitters exhaustively map supported LIR forms. An unsupported form returns an error; it never falls back to `i64`, `float`, or another convenient representation.

The LLVM target pipeline uses `mlir-opt-21`, `mlir-translate-21`, and
`clang-21`. Embedders may override these with `SEVERIAN_MLIR_OPT`,
`SEVERIAN_MLIR_TRANSLATE`, and `SEVERIAN_CLANG`.

Runtime operations such as string concatenation, byte allocation, ownership, errors, collections, and I/O belong behind a Severian runtime ABI. Emitters call that ABI rather than growing a private runtime in rendering functions.
