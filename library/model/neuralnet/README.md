# model.neuralnet

Import this submodule with the familiar short alias:

```sev
from model import neuralnet as nn
```

The initial layer vocabulary is `Module`, `Linear`, `LayerNorm`,
`MultiheadAttention`, and `TransformerEncoderLayer`. Layers own ordinary
Severian tensor values and expose `forward`, keeping their data flow visible to
the ownership checker and MLIR lowering pipeline.
