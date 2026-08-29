# Tensor

API ID: `library.tensor`

The package record inventories every top-level generic storage, shape, operation, composition, conversion, gradient, constructor, and legacy vector helper. Structural compiler identities remain the twelve operations under `api/compiler/tensor/`.

```sev
import tensor

def tensor_relu[T: tensor.TensorElement, *S: tensor.Dim](value: Tensor[T, *S]) -> Tensor[T, *S]:
    return tensor.relu(value)
```

Current weakness: not every operation × representation × rank × backend cell executes; illegal cells must fail before MLIR emission.
