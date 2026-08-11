# Qwen2.5-3B same-GPU benchmark

All implementations consume `models/Qwen2.5-3B-Instruct` at revision
`14d7620ba47cf51be0b176e14e27e38a34d4ff88`. Run `download_model.py` once;
never copy or convert the checkpoint per framework.

`prepare_inputs.py` writes the common 128/512/2048-token compute workloads.
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
autoregressive KV-cache decoding is still under development: tokenize `Roses
are red,`, execute every one of the 36 layers, project the final prompt position
to logits, and greedily select one token. Every valid backend must return token
348 (` v`). This is a prefill/next-token latency comparison, not the 128-token
generation benchmark described above.

The raw three-process samples are in `results/severian_example.json` and
`results/pytorch_example.json`; `results/full_pass_example_summary.json` records
their medians and explicit reasons unavailable backends have no numeric row.

Run the completed backends with:

```text
python3 benchmarks/inference/harness/run.py --output benchmarks/inference/results/severian_example.json --repetitions 3 --timeout 1800 python3 benchmarks/inference/severian/qwen.py --source benchmarks/inference/severian/benchmark.sev

python3 benchmarks/inference/harness/run.py --output benchmarks/inference/results/pytorch_example.json --repetitions 3 --timeout 1800 benchmarks/inference/.venv-pytorch/bin/python benchmarks/inference/pytorch/qwen.py --model benchmarks/inference/models/Qwen2.5-3B-Instruct --inputs benchmarks/inference/example_inputs.json --length example --end-to-end
```
