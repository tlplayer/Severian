# Transformer ROCm and PyTorch baseline

Run from the repository root:

```sh
python3 bench/transformer-rocm/run.py --chip gfx1100 --samples 20
```

The runner checks that the `with gpu:` example reaches outlined `gpu.module`
kernels with a matching `#rocdl.target`, then compares fresh-process execution
of the current Severian CPU native binary and an equivalent float64 PyTorch CPU
program. This deliberately does not publish a Severian-versus-PyTorch GPU speed
claim: executable ROCm linking and explicit tensor transfers are still absent.
