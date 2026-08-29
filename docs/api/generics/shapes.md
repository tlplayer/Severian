# Shape packs

API ID: `generic.shape`

A shape pack is an ordered sequence of zero or more dimensions. Its length is
the rank. In the current tensor library source, variadic dimensions are written
as `*S: tensor.Dim`.

```sev
import tensor

def preserve[T: tensor.TensorElement, *S: tensor.Dim](
    value: Tensor[T, *S],
) -> Tensor[T, *S]:
    return value
```

Named dimensions preserve equality across parameters and results. Constraints
such as `D % 8 == 0` can be proven statically or retained as narrow launcher
guards. They do not require a JIT merely to rediscover statically available
rank.

Current weakness: the complete source spelling and dependency-interface
contract for shape packs is not yet conformance-tested across packages.
