# Views and layout transformations

API IDs: `tensor.reshape_view`, `tensor.permute`, `tensor.broadcast`

Reshape, materialize, axis permutation, reversal, broadcast-like, and repeat
describe logical indexing/layout changes. A legal view may alias input storage;
materialization is explicit.

```sev
import tensor

def heads[T: tensor.TensorElement](value: Tensor[T, 2, 128, 1024]):
    shaped = tensor.reshape(value, [2, 128, 16, 64])
    return tensor.permute(shaped, [0, 2, 1, 3])
```

View legality depends on sizes, strides, layout, and alias/mutation constraints.
Broadcast uses projected indexing and does not imply physical duplication.
Rank and axis identity must be known where indexing maps depend on them.

Current weakness: non-contiguous legality and GPU indexed materialization are
not complete across all layouts.
