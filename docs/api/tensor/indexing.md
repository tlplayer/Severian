# Indexing and assembly

API IDs: `tensor.slice`, `tensor.gather`, `tensor.scatter`,
`tensor.concatenate`

```sev
import tensor

def select_and_join[T: tensor.TensorElement, *S: tensor.Dim, *I: tensor.Dim](
    value: Tensor[T, *S],
    indices: Tensor[i64, *I],
):
    selected = tensor.gather(copy value, indices)
    return tensor.concatenate(copy selected, selected, [0])
```

Slice carries starts, limits, and strides. Gather carries indices and axis
semantics. Scatter is effectful: mutation, collisions, aliasing, and atomic
policy must be explicit. Concatenate requires non-concatenated dimensions to
agree.

GPU lowerings must generate correct masks for dynamic bounds. Current weakness:
those mask, collision, and atomic paths are not yet implemented end to end.
