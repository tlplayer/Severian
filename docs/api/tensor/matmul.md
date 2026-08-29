# Rank-generic matmul

API ID: `tensor.matmul`

Matmul carries operand/result shapes, batch dimensions, and contraction
dimensions. Batch rank is data. There is no `matmul_rank4`, `matmul_bf16`, or
equivalent source/backend operation identity.

```sev
import tensor

def project[T: tensor.TensorElement, Batch: tensor.Dim, M: tensor.Dim, K: tensor.Dim, N: tensor.Dim](
    left: Tensor[T, Batch, M, K],
    right: Tensor[T, Batch, K, N],
) -> Tensor[T, Batch, M, N]:
    return tensor.matmul(left, right)
```

Operand ranks and contraction identities must be known before structural
emission. Batch and contraction extents may be dynamic but must be proven or
guarded compatible. Accumulator representation is target policy, not operation
identity.

Current weakness: optimized GPU contraction scheduling has not yet reached
verified ROCDL/NVVM execution through the Severian-owned pipeline.
