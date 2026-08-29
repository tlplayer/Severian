# Runtime specialization and launcher contract

Stable IDs: `compiler.runtime.*`.

`StorageViewAbi` carries pointer, element representation, rank, dimensions,
strides, offset, and ownership. It is the external storage boundary; a native
pointer is never declared as an MLIR builtin tensor result.

`CompileRegion` retains symbolic element/shape/layout/alias/effect contracts.
`KernelSpecialization` adds concrete shape, stride, and target facts. A launcher
then packs arguments, resolves the cache entry, and executes the verified
artifact.

Generic substitution (`T -> bf16`) and runtime specialization (`shape ->
[1,16,512,128]`) are separate. Neither produces names such as `load_bf16`,
`matmul_rank4`, or dtype/rank-specific operation IDs.

Only unresolved rank requires Tensor-JIT. Ranked tensors with dynamic extents
use ordinary runtime dimension operands and do not require source-level JIT
variants.

## Compiled MLIR libraries

Non-tensor library behavior follows the same ownership rule: Severian owns an
MLIR operation graph, not an opaque C implementation. A registered library has
a stable ID, ABI version, checked-in MLIR module, exports, and external platform
dependencies. Composition is demand-driven:

1. Ordinary lowering emits an unresolved ABI declaration and calls it.
2. The registry selects the library that owns that symbol.
3. The compiler parses and verifies the library module.
4. It clones requested definitions and required external declarations into the
   host module.
5. The combined module is verified before LLVM translation.

`core.text.string` version 1 is the first implementation. Its initial exports
preserve the old source String layout while `StringAbiV1` becomes the canonical
owned descriptor. This compatibility layer is explicitly transitional; it does
not authorize new runtime behavior in C.
