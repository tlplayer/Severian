# Tensor API

The tensor API has one type constructor and twelve structural compiler
operations. RMSNorm, SiLU, Softmax, RoPE, attention, and model layers are `.sev`
compositions of these primitives; they are not compiler operation IDs.

| Section | Structural IDs | Status |
| --- | --- | --- |
| [Tensor type and shape contract](type-and-shape.md) | `type.tensor` | partial unranked execution |
| [Elementwise](elementwise.md) | `tensor.elementwise` | CPU partial, GPU experimental slice |
| [Reductions](reductions.md) | `tensor.reduce` | partial |
| [Matmul](matmul.md) | `tensor.matmul` | partial |
| [Views, reshape, permute, broadcast](views-and-layout.md) | `tensor.reshape_view`, `tensor.permute`, `tensor.broadcast` | partial |
| [Slice, gather, scatter, concatenate](indexing.md) | four structural IDs | partial |
| [Conversion](conversion.md) | `tensor.convert` | partial representation matrix |
| [Storage view and specialization](storage-view.md) | `tensor.storage_view` | partial runtime route |

Machine records are in
[`../compiler/tensor/operations.toml`](../compiler/tensor/operations.toml), and
the exhaustive executable goal is
[`../../examples/08-numerics/15-tensor-exhaustive.sev`](../../examples/08-numerics/15-tensor-exhaustive.sev).

## Non-negotiable invariants

- Element type, rank, dimensions, strides, layout, aliasing, mutation, and
  runtime shape operands are IR/ABI data.
- Rank zero and unranked are distinct.
- Dynamic dimension and unknown rank are distinct.
- Rank-2 and rank-4 matmul share `tensor.matmul`.
- `Tensor[f16]`, `Tensor[bf16]`, and `Tensor[f32]` share operation identities.
- Storage pointers never masquerade as MLIR builtin tensors.
