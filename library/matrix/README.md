# matrix

`matrix` is the symbolic linear-algebra layer beneath `tensor`. Its decorator
pack imports mathematical notation into library implementations:

```sev
import matrix

@matrix(X, ^, I, J)
def algebra(A: Matrix[f32], B: Matrix[f32], u: list[float], v: list[float]):
    product = A X B
    normal = u ^ v
    basis = I(4)
    jacobian = J([1.0, 0.0, 1.0])
```

`X` is matrix multiplication, `^` is the three-dimensional cross product, `I`
constructs an identity matrix, and `J` builds a row-major diagonal Jacobian
from derivative values. These are namespace symbols, not Python decorators.

The current `Matrix` intrinsic records shape and a uniform fill value. Ranked,
contiguous element storage is the next representation step; that representation
will lower through MLIR tensor/linalg operations and become the storage layer
used by `tensor`.
