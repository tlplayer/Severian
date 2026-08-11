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
