# Popcorn kernel export

Popcorn submissions are single Python files. Severian therefore exports a thin
problem adapter rather than submitting a standalone process:

```text
typed Severian tensor function
        ↓
sev kernel export popcorn
        ↓
submission.py with a generated Triton kernel
        ↓
Torch-owned device pointers and current CUDA/HIP stream
```

The initial backend supports GPU Mode leaderboard 544, `vectorsum_v2`. Given a
one-input tensor function returning `tensor.sum` or `tensor.rankedSum`:

```sh
sev kernel export popcorn kernel.sev \
    --entry reductionSum \
    --leaderboard vectorsum_v2 \
    --gpu A100 \
    --output submission.py

popcorn submit submission.py --mode benchmark
```

The generated file defines Popcorn's required `custom_kernel`, embeds provenance
for the selected Severian entry and operation, and launches Triton directly on
the Torch allocations. It performs no CPU copy, serialization, or per-invocation
process spawn. Triton uses Torch's active CUDA or HIP stream.

`--block-size` accepts a power of two from 64 through 65536 and defaults to
1024. It is exposed so reductions can be tuned locally and then regenerated.

Other leaderboard contracts currently fail with an explicit unsupported error.
They need dedicated adapters because each Popcorn problem defines its own input
tuple, output ownership, shape cases, and correctness contract. A general native
Severian/Torch ABI still requires external device-buffer adoption and current
stream interoperation; the existing PJRT runtime does not claim those semantics.

The submission contract follows the official
[Popcorn CLI format](https://github.com/gpu-mode/popcorn-cli#submission-format),
and the adapter matches the open-source
[`vectorsum_py` problem](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/vectorsum_py).
