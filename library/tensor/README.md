# tensor

`tensor` provides portable, shape-oriented kernels used by numerical and
machine-learning packages. The initial executable implementation uses flat
`list[float]` storage, explicit shapes, ordinary loops, and Severian tasks.
That makes ownership and bounds behavior visible while the compiler grows a
first-class ranked tensor type.

The public operations currently include elementwise ReLU, task-parallel ReLU,
matrix-vector multiplication, vector addition, and a flattened ReLU Jacobian.

## Lowering contracts

Consumers may annotate a kernel with one of these tensor symbol-pack policies:

```sev
@tensor(SIMD)
@tensor(GPU)
@tensor(AUTO)
```

The decorators are retained in HIR and emitted as a
`severian_tensor_policy` MLIR function attribute. `SIMD` is intended to select
MLIR vector/LLVM-vector lowering, `GPU` is intended to select MLIR GPU plus
NVVM/ROCDL lowering, and `AUTO` will use shape and target cost models. Until
those passes exist, all policies execute the portable CPU implementation; the
compiler must not claim that CUDA code was generated.

The API keeps execution policy separate from mathematical behavior, allowing
`neuralnet` to share activation and Jacobian definitions across CPU, SIMD, and
GPU targets.
