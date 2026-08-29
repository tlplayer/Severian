# Reductions

API ID: `tensor.reduce`

Kinds currently include sum, sum-axis, mean-last, and max-last. Reduction axes
are structural data, not separate operation identities.

```sev
import tensor

def row_total[T: tensor.TensorElement, Rows: tensor.Dim, Columns: tensor.Dim](
    value: Tensor[T, Rows, Columns],
):
    return tensor.sum_axis(value, 1)
```

Rank and axis identities must be known. Axis extents may be dynamic. The
lowering chooses a representation-correct identity, combiner, accumulator, and
optional finalizer. Mean is sum plus a width-dependent finalizer; RMSNorm and
Softmax compose reductions with elementwise operations.

Current weakness: GPU scheduling and the complete accumulator policy are not
implemented end to end.
