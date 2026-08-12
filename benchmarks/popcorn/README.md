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
  --output target/popcorn/reduction_sum.triton.py

python3 benchmarks/popcorn/bundle.py \
  target/popcorn/reduction_sum.triton.py \
  benchmarks/popcorn/vectorsum_v2/adapter.py \
  target/popcorn/submission.py \
  --leaderboard vectorsum_v2 --gpu A100

popcorn submit target/popcorn/submission.py \
  --leaderboard vectorsum_v2 --gpu A100 --mode benchmark
```

The generated Triton module contains `launch`, compiler metadata, and the
device kernel. It contains no `custom_kernel`, task imports, leaderboard names,
or Popcorn directives. `bundle.py` only satisfies the harness's current
single-file submission rule.

`sev kernel inspect` explains the selected backend and fallback. `sev kernel
emit --backend xla` emits StableHLO for the same Severian entry, which makes it
possible to compare the portable and specialized paths without changing the
source program.
