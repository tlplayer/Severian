# Elementwise operations

API ID: `tensor.elementwise`

Kinds currently include add, subtract, multiply, divide, exp, log, sin, tanh,
rsqrt, relu, scale, and add-scalar. Kind is an attribute of one operation ID.

```sev
import tensor

def residual[T: tensor.TensorElement, *S: tensor.Dim](
    hidden: Tensor[T, *S],
    update: Tensor[T, *S],
) -> Tensor[T, *S]:
    return tensor.add(hidden, update)
```

Rank must be known when generating rank-dependent maps; dimensions may remain
dynamic. Floating and integer operations choose representation-correct scalar
MLIR from the resolved element field. Fusion may remove intermediate storage
without changing ownership or alias semantics.

Current weakness: GPU lowering covers an initial blocked elementwise slice, not
the complete operation/representation/layout matrix.
