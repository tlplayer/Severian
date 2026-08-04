# Transformer encoder on ROCm

This example uses `with gpu:` around a complete transformer encoder forward and
training step. It includes scaled dot-product attention, softmax, residuals,
layer normalization, a ReLU FFN, reverse-mode gradient kernels, and SGD weight
updates. The compiler outlines the ranked tensor operations, serializes gfx11
ROCDL code objects, links the HIP runtime, and launches them on AMD GPUs.

```sh
sev compile main.sev --target rocm --chip gfx1101 -o /tmp/transformer
SEVERIAN_ROCM_TRACE=1 /tmp/transformer
```

When `--chip` is omitted, the driver checks `SEVERIAN_AMDGPU_CHIP`,
`amdgpu-arch`, `rocminfo`, and known AMD PCI IDs from `lspci`.
`for i in indices(values) with gpu:` is the
single-loop form of the same explicit placement. Replace `gpu` with `simd` for
a host-vector execution region.
