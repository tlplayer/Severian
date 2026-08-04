# Transformer-style ROCm lowering

This example implements the affine-plus-bias-plus-activation portion of a
transformer feed-forward block with ranked tensors. `with gpu:` is the explicit
parallel execution boundary. A single loop may equivalently use
`for i in indices(values) with gpu:`; the parser represents both forms as the
same HIR region. Replace `gpu` with `simd` to request a host-vector region.

Inspect ordinary MLIR or lower the parallel linalg kernels through GPU dialect
outlining to AMD ROCDL:

```sh
sev emit-mlir main.sev
sev emit-mlir main.sev --target rocm --chip gfx1100
```

When `--chip` is omitted, the driver uses `SEVERIAN_AMDGPU_CHIP` or an installed
`amdgpu-arch`. The target form currently emits target-specific ROCDL MLIR; it
does not yet link a runnable HIP executable or insert host/device transfers.
