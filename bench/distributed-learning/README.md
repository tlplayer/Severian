# Distributed learning comparison

This comparison runs the same 65,536-value ReLU forward and backward pass over
four workers:

- Severian uses imported `distributed`, `tensor`, and `neuralnet` packages and
  native pthread-backed tasks.
- Python uses `concurrent.futures.ProcessPoolExecutor` and ordinary lists.
- PyTorch uses one tensor, four CPU threads, `torch.relu`, and autograd.

Both programs must exactly match the Severian stdout fixture before a sample is
accepted. Run:

```sh
python3 bench/distributed-learning/run.py
python3 bench/distributed-learning/run.py --samples 30 --warmup 5
```

The default PyTorch interpreter is `/tmp/severian-onnx-venv/bin/python` during
development. Pass `--torch-python /path/to/python` to select any environment
containing PyTorch.

This measures a coarse-grained end-to-end process, including imports and worker
startup. It is useful for deployment/API overhead comparison, not as evidence
of GPU or SIMD throughput. The Severian `local` backend shares read-only tensors
between threads; Python serializes shard lists to worker processes; PyTorch
executes vectorized native tensor kernels and derives the backward pass with
autograd. The report also shows a warm PyTorch tensor/autograd call after the
framework is loaded, so import time is not mistaken for kernel performance.
