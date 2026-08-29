# Storage view and runtime specialization

API ID: `tensor.storage_view`

Host storage crosses a versioned descriptor boundary containing data pointer,
element representation, rank, dimensions, strides, offset, layout, ownership,
and alias metadata. A native pointer is never declared as an MLIR tensor.

```sev
import tensor

def load_weight[T: tensor.TensorElement](entry: tensor.SafeTensorEntry) -> Tensor[T]:
    return tensor.load[T](entry)
```

For a concrete invocation, runtime metadata becomes a kernel specialization:

```text
StorageViewAbi
  → shape/rank/stride specialization
  → specialized CPU MLIR or Severian GPU MLIR
  → compiled-kernel cache
  → launcher
  → execution
```

The cache key includes graph hash, concrete shape/strides, element
representation, architecture, compiler revision, and options. This is runtime
specialization, not dtype/rank-specific source functions.

Current weakness: the descriptor-to-cached-launcher route is not yet fully
connected for unranked external storage. Legalization correctly rejects a
rank-dependent region before malformed MLIR is emitted.
