# Popcorn benchmark adapters

Popcorn is an external correctness and performance harness, not a Severian
compiler backend. Severian emits a reusable kernel artifact; the files here
adapt that artifact to an individual benchmark's Python protocol.

For the vector-sum benchmark:

```sh
cargo build -p severian-driver

target/debug/sev kernel inspect \
  benchmarks/popcorn/vectorsum_v2/kernel.sev \
  --entry reduction_sum --target gpu

target/debug/sev kernel emit \
  benchmarks/popcorn/vectorsum_v2/kernel.sev \
  --entry reduction_sum --backend triton \
  --output target/popcorn/reduction_sum.ttir.mlir
```

The generated artifact is native Triton MLIR. It contains the device kernel and
launch metadata, with no Python or Torch dependency. The legacy `bundle.py`
and `vectorsum_v2/adapter.py` files remain only as documentation of Popcorn's
Python submission protocol; they cannot execute TTIR and are not part of the
compiler path. A native Popcorn runner must compile the TTIR and bind its
pointer/count ABI before submission.

`sev kernel inspect` explains the selected backend and fallback. `sev kernel
emit --backend xla` emits StableHLO for the same Severian entry, which makes it
possible to compare the portable and specialized paths without changing the
source program.
