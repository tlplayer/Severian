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
      +-- Triton frontend --> TritonIR --> GPU kernel
      +-- LLVM dialect --> native CPU code
```

StableHLO/XLA is the portable tensor path. Direct Triton lowering is selected
only for recognized GPU regions with a supported specialized lowering. XLA is
retained as the fallback. CPU code continues through the native LLVM path.

Backend selection is inspectable:

```sh
sev kernel inspect source.sev --entry reductionSum --target gpu
```

An automatic GPU selection reports the recognized kernel operation, selected
backend, fallback, and reason. Explicit `--backend xla`, `--backend triton`, and
`--backend llvm` options are policy overrides for diagnosis and benchmarking;
they do not change Severian language semantics.

Kernel artifacts are emitted without a benchmark-specific ABI:

```sh
sev kernel emit source.sev --entry reductionSum --backend triton \
  --output reduction_sum.triton.py

sev kernel emit source.sev --entry reductionSum --backend xla \
  --output reduction_sum.stablehlo.mlir
```

The initial specialized operation is tensor reduction sum. The emitted Python
module is a thin carrier for a generated Triton kernel and exports `launch`.
The Triton frontend lowers that kernel to TritonIR for the installed GPU
toolchain. Future lowering can replace this carrier with serialized TritonIR
without changing kernel IR, selection, or benchmark adapters.

Popcorn integration lives under `benchmarks/popcorn`. It consumes emitted
artifacts just like any other external harness and is intentionally absent from
the compiler and language.
