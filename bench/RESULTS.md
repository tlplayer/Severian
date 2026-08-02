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

Severian native compilation took 160.308 ms.

| Engine | Median fresh process (ms) | p95 fresh process (ms) |
| --- | ---: | ---: |
| Severian, four local shards | 146.372 | 153.009 |
| Severian, sequential control | 315.215 | 325.097 |
| PyTorch | 1,049.215 | 1,056.340 |
| ONNX Runtime | 162.220 | 165.954 |

| Warm engine call | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| PyTorch | 1.703 | 1.948 |
| ONNX Runtime | 0.414 | 1.142 |

Every engine produced 180,000 logits with exact class counts
`[20000, 20000, 20000]`; the runner also checked per-class logit checksums
within its floating-point tolerance. The four-shard Severian executable was
about 2.15x faster than its sequential control. Its complete fresh-process time
was in the same range as ONNX Runtime's fresh process, while the warm batched
PyTorch and ONNX Runtime calls show the substantial kernel-performance gap that
remains.

## What the result says

The experiment already validates several useful language properties:

- A real ONNX model can be translated into Severian source and checked against
  the same weights running in PyTorch and ONNX Runtime.
- Scoped task ownership and `local` placement provide a measurable parallel
  speedup without hiding shard boundaries or join order.
- The `models` symbol pack now lets source say `Relu(X)` and `J(X)`, while the
  scalar package definition uses the piecewise expression
  `0.0 if x < 0.0 else x`.

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
- Seven measured fresh-process samples after two warmups
- CPU execution only; no CUDA or SIMD backend is claimed

Generated ONNX data and binaries are intentionally ignored. Preparation and
dependency commands are documented in [onnx-gold/README.md](onnx-gold/README.md).
