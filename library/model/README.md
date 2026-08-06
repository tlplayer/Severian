# model

`model` is the author-facing tensor namespace. It uses conventional operation
names—`tensor`, `matmul`, `add`, `relu`, `softmax`, `transpose`, and
`layerNorm`—while keeping Severian's runtime and compiler independent from any
particular Python framework.

```sev
import model
from model import neuralnet as nn

X = model.tensor([1.0, 2.0], [1, 2])
linear = nn.Linear(model.tensor([1.0, 0.0, 0.0, 1.0], [2, 2]), model.tensor([0.0, 0.0], [2]))
Y = linear.forward(X)
```

The lower-level `tensor` and `models` packages remain available for compiler
fixtures, graph construction, autodiff experiments, and backend work.
