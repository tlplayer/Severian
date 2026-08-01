# Distributed learning comparison

This comparison runs the same 65,536-value ReLU forward and backward pass over
four workers:

- Severian uses imported `distributed`, `tensor`, and `neuralnet` packages and
  native pthread-backed tasks.
- Python uses `concurrent.futures.ProcessPoolExecutor` and ordinary lists.

Both programs must exactly match the Severian stdout fixture before a sample is
accepted. Run:

```sh
python3 bench/distributed-learning/run.py
python3 bench/distributed-learning/run.py --samples 30 --warmup 5
```

This measures a coarse-grained end-to-end workload, including worker startup.
It is useful for API and runtime-overhead comparison, not as evidence of GPU or
SIMD throughput. The Severian `local` backend shares read-only tensors between
threads; Python serializes shard lists to worker processes.
