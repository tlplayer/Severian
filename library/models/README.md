# models

`models` is the notation layer above `tensor`. A decorator such as
`@models(Relu, J)` imports mathematical spellings into one function's syntax
namespace; it does not wrap that function or choose an execution device.

```sev
import models

@models(Relu, J)
def activationPass(X: list[float]):
    values = Relu(X)
    jacobian = J(X)
```

`Relu` resolves to `models.activation`, and `J` resolves to the ReLU activation
Jacobian. `X` is an ordinary local binding here, following the conventional
name for model input. The infix matrix-product symbol `X` remains owned by the
separate `math` package and is enabled with `@math(X)`.

The same pack can expose `LeakyRelu`, `FastSigmoid`, `FastTanh`, `Gelu`, and
`Swish`. The `Fast` names are intentional: those two tensor kernels are the
current inexpensive rational approximations, not the exact transcendental
functions.

The scalar definitions deliberately use Severian's conditional expression:

```sev
0.0 if x < 0.0 else x       # ReLU(x)
0.0 if x <= 0.0 else 1.0    # its chosen derivative at x = 0
```

The Jacobian is returned as a flattened row-major diagonal matrix until ranked
tensors become a first-class language type.

## Compiled model graphs

Ranked tensor models can be captured before execution. `graphInput` introduces
an eager tensor, graph operations build symbolic dependencies, and `runGraph`
is the single execution boundary:

```sev
import models
import tensor

with gpu:
    X = models.graphInput(input)
    W = models.graphInput(weights)
    outputNode = models.graphRelu(models.graphMatmul(X, W))
    output = models.runGraph(outputNode)
```

The package manifest assigns semantic roles to these operations. The compiler's
model-graph pass uses those roles for common-subexpression elimination without
hard-coding the `models` spellings. Graph construction also removes a double
transpose, drops an identity scale, and combines adjacent scales.

On ROCm, all forward nodes execute on one HIP stream and synchronize once at
`runGraph`; the eager tensor operations previously created and synchronized a
stream for every node. Execution still records the tensor autodiff tape, so
`tensor.rankedBackwardMse(output)` and SGD work after the graph run. The bundled
`transformerEncoderGraph` builds single-head attention, softmax, residuals, two
layer normalizations, and a ReLU feed-forward network.
