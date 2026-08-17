# Transformer container

This vertical example uses the author-facing model vocabulary:

```sev
import model
from model import neuralnet as nn

tokens = model.tensor(values, [sequenceLength, hiddenSize])
encoder = nn.TransformerEncoderLayer(attention, linear1, linear2, nn.LayerNorm(), nn.LayerNorm())
output = encoder.forward(tokens)
```

It executes scaled dot-product attention, residual connections, two layer
normalizations, and a `2 -> 4 -> 2` feed-forward network using fixed weights.
The small shape is a deterministic correctness and launch-overhead workload,
not a claim about large-model throughput.

Build the packages first and then run the produced executable:

```sh
cd docs/examples/28-transformer-container
sev build
./target/debug/transformer-container-example
```

Build the exact executable into an OCI image with Podman or Docker:

```sh
podman build -t severian/transformer:local -f Containerfile .
podman run --rm --network none --memory 256m --cpus 1 severian/transformer:local
```

The image runs as an unprivileged numeric user. The example's Severian test
also validates the matching namespace, CPU, memory, network, and read-only
root filesystem plan from the `container` package.

For repeatable host-versus-container measurements, see
`bench/transformer-container`.
