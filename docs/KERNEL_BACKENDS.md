# Kernel backend architecture

Severian keeps language semantics separate from backend policy:

```text
Severian source
      |
      v
typed HIR
      |
      v
Severian kernel IR
      |
      +-- StableHLO --> XLA/PJRT
      +-- native TTIR --> Triton MLIR compiler --> GPU kernel
      +-- LLVM dialect --> native CPU code
```

StableHLO/XLA is the portable tensor path. Direct Triton lowering is selected
only for recognized GPU regions with a supported specialized lowering. XLA is
retained as the fallback. CPU code continues through the native LLVM path.

Backend selection is inspectable:

```sh
sev kernel inspect source.sev --entry reduction_sum --target cuda:sm_90
```

Source may request a policy without importing a compiler package:

```sev
@compile(triton)
def reduction_sum(value: Tensor[f32, dynamic]) -> Tensor[f32]:
    return tensor.sum_last_f_32(value)
```

`@compile(auto)`, `@compile(xla)`, and `@compile(triton)` are built-in compiler
policy decorators. A command-line `--backend` option overrides the source
policy for comparison and diagnosis.

An automatic GPU selection reports the recognized kernel operation, selected
backend, fallback, and reason. Explicit `--backend xla`, `--backend triton`, and
`--backend llvm` options are policy overrides for diagnosis and benchmarking;
they do not change Severian language semantics.

Kernel artifacts are emitted without a benchmark-specific ABI:

```sh
sev kernel emit source.sev --entry reduction_sum --backend triton \
  --target cuda:sm_90 --output reduction_sum.ttir.mlir

sev kernel emit source.sev --entry reduction_sum --backend xla \
  --output reduction_sum.stablehlo.mlir
```

The initial specialized operations are tensor reduction sum and elementwise
ReLU. Severian emits Triton's MLIR dialect directly: `tt.func`, program IDs,
masked pointer loads/stores, reductions, and atomics. The artifact carries
launch metadata as MLIR module attributes. It contains no generated Python,
Torch import, or Python-side Triton launcher.

Automatic selection is hardware-aware. Concrete targets such as `cuda:sm_90`
and `rocm:gfx1100` enable the specialized route. Generic `gpu`, `nvidia`, and
`amd` targets stay on StableHLO/XLA because they do not assert a compatible
architecture. NVIDIA architectures below `sm_80` also stay on XLA. An explicit
`@compile(triton)` may emit portable TTIR when the architecture is unspecified,
but a known unsupported architecture is rejected instead of producing a
misleading artifact.

External harness integration remains outside the compiler and language. A
harness must compile TTIR and bind the emitted pointer/count ABI; legacy Python
submission adapters cannot consume these artifacts directly.
