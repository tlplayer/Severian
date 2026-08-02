# Severian neural-network benchmark results

Measured 2026-08-01 on an AMD Ryzen 7 8700F (8 cores, 16 threads), using
Severian's native LLVM/MLIR backend on CPU. These results are correctness-gated:
the runner rejects a timing sample when output differs from its reference.

## Large distributed ReLU forward/backward

This benchmark applies ReLU and its backward mask to 65,536 values over four
workers. Severian uses pthread-backed `with self and local:` tasks, Python uses
`ProcessPoolExecutor`, and PyTorch uses four CPU threads plus autograd.

Command:

```sh
python3 bench/distributed-learning/run.py --samples 7 --warmup 2
```

| Implementation | Compile (ms) | Median fresh process (ms) | p95 fresh process (ms) |
| --- | ---: | ---: | ---: |
| Severian native | 125.208 | 9.296 | 9.800 |
| Python multiprocessing | 24.319 | 112.663 | 118.167 |
| PyTorch | 24.004 | 1,013.052 | 1,045.788 |

PyTorch's warm tensor/autograd call, after Python and the framework were already
loaded, measured 0.061 ms median and 0.349 ms p95.

The important split is startup versus compute. Severian's complete fresh
executable was about 12.1x faster than the equivalent fresh Python
multiprocessing program. PyTorch's fresh process is dominated by importing and
initializing the framework; its warm vectorized kernel is much faster than the
current Severian scalar/list path. All three implementations produced the exact
checked stdout fixture.

See [distributed-learning/README.md](distributed-learning/README.md) for the
workload and interpretation rules.

## Exported ONNX gold model

This is the stronger model test. It trains a real Iris classifier, exports the
model to ONNX, verifies the graph is `Gemm -> Relu -> Gemm`, and generates an
equivalent native Severian program from the ONNX initializers. The model is a
`4 -> 12 -> 3` MLP with 98.667% training accuracy. The 150 Iris observations
are repeated to make 60,000 inference rows.

Command:

```sh
python3 bench/onnx-gold/run.py --samples 7 --warmup 2
```

Severian native compilation took 178.909 ms.

| Engine | Median fresh process (ms) | p95 fresh process (ms) |
| --- | ---: | ---: |
| Severian, four shards | 156.449 | 159.287 |
| Severian, sequential | 333.924 | 343.545 |
| PyTorch | 1,153.231 | 1,168.430 |
| ONNX Runtime | 182.120 | 192.292 |

| Warm engine call | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| PyTorch | 1.892 | 2.211 |
| ONNX Runtime | 0.314 | 0.948 |

Every engine produced 180,000 logits with exact class counts
`[20000, 20000, 20000]`; the runner also checked per-class logit checksums
within its floating-point tolerance. Four local shards were about 2.13x faster
than the sequential Severian control. This benchmark does not claim affine
fusion: it measures distribution of the generated scalar/list model while the
separate activation benchmark isolates automatic fusion.

## Automatic activation-chain fusion

This benchmark applies `Relu -> FastTanh -> Swish` to 262,144 values. One source
nests the model calls, allowing a single compiler-created elementwise traversal.
The control stores each result in a binding, intentionally materializing three
traversals. Neither source contains fusion or hardware-placement syntax.

Command:

```sh
python3 bench/activation-fusion/run.py --samples 15 --warmup 3
```

| Form | Compile (ms) | Median fresh process (ms) | p95 fresh process (ms) |
| --- | ---: | ---: | ---: |
| Automatic nested fusion | 159.710 | 22.357 | 25.190 |
| Materialized control | 136.110 | 37.574 | 39.223 |

Automatic fusion was 1.681x faster in this run. Both executables produced the
exact checked stdout fixture. The compiler currently emits a CPU implementation
and records SIMD/SIMT/GPU as future lowering candidates; this result makes no
GPU or vector-backend claim.

## What the result says

The experiment already validates several useful language properties:

- A real ONNX model can be translated into Severian source and checked against
  the same weights running in PyTorch and ONNX Runtime.
- Scoped task ownership and `local` placement provide a measurable parallel
  speedup without hiding shard boundaries or join order.
- The `models` symbol pack lets source say `Relu(X)` and `J(X)`, while the
  scalar package definition uses the piecewise expression
  `0.0 if x < 0.0 else x`.
- Nested compatible activation calls can be fused without user optimization
  syntax, while explicit materialization remains visible and testable.

It also identifies the next optimization target. The Severian model currently
uses flat dynamic lists and scalar loops rather than a batched GEMM. Indexed
list values are dynamically boxed, and millions of scalar operations allocate
boxed float results. First-class ranked tensors, unboxed loop values, and an
MLIR linalg/vector lowering should therefore come before a CUDA claim. Those
changes can preserve the visible model notation and distribution contract while
replacing the expensive scalar runtime path.

## Reproduction environment

- AMD Ryzen 7 8700F, 8 cores / 16 threads
- Rust 1.96.1
- Clang/LLVM 21.1.8
- Python 3.14.4
- PyTorch 2.13.0, ONNX 1.22.0, ONNX Runtime 1.28.0, NumPy 2.3.5
- Distributed result: seven measured samples after two warmups
- ONNX result: seven measured samples after two warmups
- Activation-fusion result: fifteen measured samples after three warmups
- CPU execution only; no CUDA or SIMD backend is claimed

Generated ONNX data and binaries are intentionally ignored. Preparation and
dependency commands are documented in [onnx-gold/README.md](onnx-gold/README.md).
