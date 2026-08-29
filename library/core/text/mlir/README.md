# String MLIR library

`string_v1.mlir` is the first Severian-owned library implementation shipped as
checked-in MLIR instead of C. The compiler parses it into MLIR operations,
verifies it, imports only requested exports, and verifies the combined module
again before backend lowering.

The current vertical slice owns the operations already emitted by ordinary
String lowering:

- `__sev_string_concat`
- `__sev_string_compare`
- `__sev_string_release`

These names are compatibility adapters for the current NUL-terminated pointer
representation. Their allocation prefix intentionally matches the former C
implementation so ownership remains coherent while the source representation
migrates to `StringAbiV1`. That legacy prefix is currently a 64-bit contract;
the legality gate rejects this library on a 32-bit target instead of silently
using the wrong layout.

Calls to `malloc`, `free`, `strlen`, `strcmp`, `memcpy`, and `abort` are external
platform ABI declarations. They are not Severian library implementations. New
String behavior belongs in MLIR (or in `.sev` compiled to MLIR), not in a C
source file.

The active migration order is:

1. Prove checked-in MLIR composition using the compatibility exports.
2. Move source String lowering to the versioned `{data, length, capacity}` ABI.
3. Replace terminator-dependent operations with length-aware MLIR operations.
4. Port the remaining conversion and formatting helpers out of `string.c`.

`examples/mlir-runtime.sev` is the source-level execution proof for this slice.
