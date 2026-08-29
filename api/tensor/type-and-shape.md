# Tensor type and shape

API ID: `type.tensor`

`Tensor[T, S...]` combines an element type with an ordered shape. A ranked
tensor may contain known and dynamic dimensions. An unranked tensor carries no
compile-time rank and cannot enter rank-dependent lowering until specialization.

```sev
import tensor

def matrix[T: tensor.TensorElement, Rows: tensor.Dim, Columns: tensor.Dim](
    value: Tensor[T, Rows, Columns],
) -> Tensor[T, Rows, Columns]:
    return value
```

Strides, layout, offset, aliasing, mutation, and ownership are part of the
value contract even when they are not all present in the source spelling.
Dynamic sizes become runtime shape operands; known rank lets emitters construct
the correct number of indexing dimensions.

Current weakness: fully opaque external storage reaches the correct
specialization boundary, but the complete cached launcher path is not yet
executable end to end.
