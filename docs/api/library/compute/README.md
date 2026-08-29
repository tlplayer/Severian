# Compute libraries

Stable IDs: `library.compute`, `library.compute.distributed`,
`library.compute.parallel`, and `library.tensor`.

The compute root owns execution-policy composition. `parallel` describes
parallel work and placement; `distributed` describes work spanning nodes or
devices; tensor owns typed multidimensional computation and its compiler
protocol. These packages depend on language concurrency/effect rules but do not
redefine them.

Tensor operations are documented under [`tensor.*`](../tensor/README.md).
RMSNorm, SiLU, Softmax, RoPE, attention, and Qwen layers are library
compositions, not compiler operation IDs. Placement selects a backend after
semantic typing; it does not alter the source operation identity.

Current uncertainty: distributed failure/retry semantics and full GPU backend
coverage are partial. Programs must not infer availability merely from package
resolution.
