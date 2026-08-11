# Inference comparison

| | PyTorch eager | PyTorch compile (warm) | Severian XLA/PJRT | SGLang (public) | MAX/Mojo (public) |
| --- | --- | --- | --- | --- | --- |
| Hardware | AMD Radeon RX 7700, `gfx1101`, 16 GB | AMD Radeon RX 7700, `gfx1101`, 16 GB | AMD Radeon RX 7700, `gfx1101`, 16 GB | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each | 1× NVIDIA A100 80 GB SXM |
| Model | Qwen2.5-3B-Instruct, BF16 | Qwen2.5-3B-Instruct, BF16 | Qwen2.5-3B-Instruct, BF16 | Qwen3.5-122B-A10B, FP8 weights/BF16 KV | Llama 3.1 8B, BF16 |
| Workload | Local 32-token full forward, batch 1 | Local 32-token full forward, batch 1 | Local 32-token full forward, batch 1; plus 256-in/32-out serving, 80 requests, concurrency 16 | External 200-in/200-out burst, 500 requests | External ShareGPTv3 burst, 500 requests |
| Warm 32-token forward | 67.834 ms median | 57.336 ms median | **50.509 ms median** | Not measured locally | Not measured locally |
| Output tok/s | 14.742 tok/s | 17.441 tok/s | **19.798 tok/s** | 1,985 tok/s | 3,860 tok/s |
| Three forward samples | 67.308 / 68.480 / 67.834 ms | 54.560 / 57.336 / 57.966 ms | 50.979 / 47.395 / 50.509 ms | — | — |
| Process to ready | 4.820 s median | 10.267 s median | 8.273 s median | — | — |
| Local serving output throughput | Not measured | Not measured | **19.747 tok/s** | Did not run on `gfx1101` | Did not run on `gfx1101` |
| Published output throughput | — | — | — | 1,985 tok/s | 3,860 tok/s |
| Theoretical RX 7700 scaling range | — | — | 19.747 tok/s measured | 12.506–96.951 tok/s | 623.538–1,181.285 tok/s |
| Comparison status | Same GPU/model/shape as Severian | Same GPU/model/shape as Severian; compilation completed during untimed warmup | Only completed local serving implementation | Different model, workload, precision, hardware, and GPU count | Different model, workload, hardware, and serving stack |
| Source | [Raw eager samples](results/pytorch_example.json) | [Raw compiled samples](results/pytorch_compile_example.json) | [Forward samples](results/severian_example.json) · [Serving requests](results/severian_serving_256x32_c16.json) | [Public result](https://github.com/sgl-project/sglang/issues/19603) · [Benchmark guide](https://docs.sglang.ai/developer_guide/bench_serving) | [Published result](https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform) · [Methodology](https://www.modular.com/blog/max-gpu-state-of-the-art-throughput-on-a-new-genai-platform) |
