# Example benchmarks

The checked neural-network measurements and their interpretation are recorded
in [RESULTS.md](RESULTS.md). The backend and autodiff work derived from those
measurements is tracked in [PARALLEL_ROADMAP.md](PARALLEL_ROADMAP.md).

This suite compares the complete Severian example inventory in directories
`00-getting-started` through `07-generics-constraints` with equivalent Rust and
Python programs. It measures fresh-process execution, not an in-process loop,
so the numbers include process startup and runtime initialization. That is the
honest measurement for these small command-line examples; it is not a language
throughput benchmark.

Run it from the repository root:

```sh
python3 bench/run.py
```

Useful options:

```sh
python3 bench/run.py --samples 100 --warmup 10 --csv bench/results.csv
python3 bench/run.py --check
tests/check_bench_examples.sh
```

The runner builds `sev` once, then measures each example's Severian and Rust
compilation separately. Python's compile column measures bytecode compilation.
Before timing execution, all three implementations must exit successfully,
write nothing to stderr, and exactly match the adjacent `.stdout` fixture.
Every timed sample is checked again. A missing fixture, missing counterpart,
compiler error, or output mismatch is reported and makes the command fail.

Generated binaries, bytecode, and temporary files live under `bench/.work/`.
Machine-specific CSV results are ignored by Git.

The separate `distributed-learning/` comparison exercises a 65,536-value
four-worker neural-network forward/backward pass against Python multiprocessing.
It also includes an equivalent PyTorch/autograd implementation.

`activation-fusion/` compares nested
`Swish(FastTanh(Relu(X)))` with an explicitly materialized control. Neither
program requests fusion or names a hardware backend; the compiler recognizes
the nested elementwise graph through the model/tensor library calls.

`onnx-gold/` trains a real Iris classifier, exports a checked ONNX graph,
generates an equivalent four-shard native Severian program from its weights,
and validates it against PyTorch and ONNX Runtime before reporting timing.
