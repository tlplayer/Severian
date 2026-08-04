# Transformer ROCm benchmark

This benchmark runs the same float64 transformer encoder forward pass and
reverse-mode training step in Severian and PyTorch on an AMD GPU. The graph has
scaled dot-product self-attention, row softmax, residual connections, two layer
normalizations, a `2 -> 4 -> 2` ReLU FFN, mean-square loss, backward kernels,
and an SGD update.

Run from the repository root:

```sh
python3 bench/transformer-rocm/run.py --chip gfx1101 --iterations 20 --warmup 3
```

Severian timing happens inside one persistent process after warmup, so ROCm
initialization, code-object loading, and compilation are excluded. The model is
captured with the `models` graph API; its compiler pass shares common nodes and
the ROCm executor uses one stream and one synchronization for the forward graph.
PyTorch uses its ROCm `cuda` surface and synchronizes around each measured sample.

The checked-in dataset is intentionally tiny (3 tokens, hidden width 2, one
head, FFN width 4) so it is a correctness and launch-overhead baseline, not a
throughput claim. The runner rejects forward or updated-weight mismatches.

Some locally packaged PyTorch ROCm builds require additional ROCm shared
libraries. Point `LD_LIBRARY_PATH` at those libraries before running; the
benchmark does not download or install dependencies itself.
