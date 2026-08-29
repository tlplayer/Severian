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
