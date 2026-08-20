# Backend boundary

The backend boundary turns LIR into artifacts or delegates LIR to a concrete backend implementation.

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
- C runtime semantics embedded as an expanding string inside the emitter.

Concrete emitters exhaustively map supported LIR forms. An unsupported form returns an error; it never falls back to `i64`, `float`, or another convenient representation.

Runtime operations such as string concatenation, byte allocation, ownership, errors, collections, and I/O belong behind a Severian runtime ABI. Emitters call that ABI rather than growing a private runtime in rendering functions.
