# Qwen2.5-3B same-GPU benchmark

All implementations consume `models/Qwen2.5-3B-Instruct` at revision
`14d7620ba47cf51be0b176e14e27e38a34d4ff88`. Run `download_model.py` once;
never copy or convert the checkpoint per framework.

`prepare_inputs.py` writes the common 128/256/512/2048-token compute workloads.
The harness rejects a sample unless it includes the GPU identity, readiness and
first-token timestamps, peak VRAM, and actual generated token IDs. Raw samples
belong under `results/`; summarized tables must be derived from those files and
must not use best-of-N selection.

Cold-load runs require 20 fresh processes. Warm and concurrency runs require at
least 100 completed requests per lifecycle/concurrency row. The Severian row is
not eligible until the complete checkpoint-backed 36-layer graph, GPU KV cache,
and greedy decode execute through StableHLO/XLA/PJRT.

## One-pass correctness and latency example

`example_inputs.json` defines the smaller full-forward-pass workload used while
autoregressive KV-cache decoding is still under development. It contains one
canonical array of exactly 32 token IDs, an all-ones attention mask, and position
IDs 0 through 31. Every backend consumes those values directly as shape `[1,
32]`, executes all 36 layers, reads logits at position 31, and must return token
3170 (` why`). Tokenization and argmax validation are outside the synchronized
forward timer. This is a prefill latency comparison, not the 128-token generation
benchmark described above.

The raw three-process samples are in `results/severian_example.json` and
`results/pytorch_example.json`; `results/full_pass_example_summary.json` records
their medians and explicit reasons unavailable backends have no local numeric
row. The old five-token PyTorch versus 32-token Severian result is superseded.

| Backend | Sequence | Median forward | Process-to-ready |
| --- | ---: | ---: | ---: |
| Severian XLA | 32 | 50.509 ms | 8.273 s |
| PyTorch eager BF16 | 32 | 67.834 ms | 4.820 s |

At this shape Severian's synchronized forward is 25.54% lower latency (1.343x),
while PyTorch reaches the ready state sooner. All six samples have fixture hash
`acfbbb0c28d4b98319d4094c52baadabb4e02fb77e9adcb5d8405f4b6feea5e3`
and return token 3170.

Run the completed backends with:

```text
python3 benchmarks/inference/harness/run.py --output benchmarks/inference/results/severian_example.json --repetitions 3 --timeout 1800 python3 benchmarks/inference/severian/qwen.py --source benchmarks/inference/severian/benchmark.sev --inputs benchmarks/inference/example_inputs.json --length 32

python3 benchmarks/inference/harness/run.py --output benchmarks/inference/results/pytorch_example.json --repetitions 3 --timeout 1800 benchmarks/inference/.venv-pytorch/bin/python benchmarks/inference/pytorch/qwen.py --model benchmarks/inference/models/Qwen2.5-3B-Instruct --inputs benchmarks/inference/example_inputs.json --length 32
```
