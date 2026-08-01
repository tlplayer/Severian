# tensor

`tensor` provides portable, shape-oriented kernels used by numerical and
machine-learning packages. The initial executable implementation uses flat
`list[float]` storage, explicit shapes, ordinary loops, and Severian tasks.
That makes ownership and bounds behavior visible while the compiler grows a
first-class ranked tensor type.

The public operations include ReLU, leaky ReLU, fast sigmoid, fast tanh, GELU,
Swish, backward kernels, activation Jacobians, task-parallel ReLU,
matrix-vector multiplication, and vector addition. The sigmoid and tanh
implementations are inexpensive rational approximations suitable for showing
portable lowering; exact transcendental variants will use math intrinsics.

## Lowering direction

These kernels currently execute through portable scalar loops and native local
tasks. Severian decorators import a package's syntax symbols; they are not
Python-style wrappers or execution-policy annotations. A future SIMD/GPU
selection construct therefore needs its own expression- or task-local syntax
and lowering semantics before this package claims either backend.

The API keeps mathematical behavior independent of execution placement, so
`neuralnet` can reuse the same activation and Jacobian definitions as those
backends are added.
