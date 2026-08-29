# Dimension parameters

API ID: `generic.dimension`

A dimension is a non-negative shape value, not a type. It may be known,
symbolic, constrained, or supplied as a runtime shape operand while rank stays
known.

```sev
import tensor

def matrix[T: tensor.TensorElement, Rows: tensor.Dim, Columns: tensor.Dim](
    value: Tensor[T, Rows, Columns],
) -> Tensor[T, Rows, Columns]:
    return value
```

Dimension expressions use checked exact arithmetic. Conflicting equalities,
overflow, division by zero, and non-exact division are constraint errors.

`Dynamic` dimension and `Unranked` tensor are different contracts. The former
permits rank-dependent emission; the latter must specialize before such
emission.

Current weakness: dimension constraints are represented in Universal, but the
full source-to-launcher guard path is not yet conformance-tested for every
dynamic expression.
