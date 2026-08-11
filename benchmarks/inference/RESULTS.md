# Qwen serving benchmark results

## Local Severian result

This is the only locally measured serving result in this report. Severian ran
Qwen2.5-3B-Instruct in BF16 through StableHLO/XLA/PJRT on the AMD Radeon RX 7700
(`gfx1101`). XLA reported the ROCm device, and every tensor function executed on
that device. One warmup request completed before measurement. The measured
workload used one fixed 256-token ID array for all requests, generated 32 tokens
greedily, submitted 80 requests in five waves of 16 in-flight requests, and kept
all 36 layers of K/V cache on the GPU. The scheduler interleaves batch-one graphs
round-robin; it does not claim continuous/dynamic batching.

Each XLA output is awaited when its token argmax is copied to the host, so TTFT,
TPOT, and the workload boundary are synchronized. Model loading and XLA
compilation are excluded from serving throughput and reported separately.

| Metric | Result |
| --- | ---: |
| Output throughput | 19.747 tok/s |
| Total throughput (input + output) | 177.722 tok/s |
| Request throughput | 0.617 req/s |
| TTFT p50 / p95 | 1,061.359 / 1,995.118 ms |
| TPOT p50 / p95 | 767.518 / 818.621 ms |
| Measured workload duration | 129.641 s |
| Process to weights mapped | 3.985 ms |
| Warmup / XLA compilation | 19,100.484 ms |
| Process to ready | 19,727.725 ms |
| Peak sampled VRAM | 14,661,783,552 bytes (13.65 GiB) |
| Physical VRAM exposed by amdgpu | 17,163,091,968 bytes (15.98 GiB) |

All 80 requests returned the same 32 token IDs. The raw JSON contains every
request's start, first-token and completion timestamp, all token timestamps,
token IDs, TTFT, end-to-end latency, and 31 inter-token intervals:
[`results/severian_serving_256x32_c16.json`](results/severian_serving_256x32_c16.json).

Reproduce it with:

```text
cargo build -p severian-xla -p severian-driver
benchmarks/inference/.venv-pytorch/bin/python benchmarks/inference/severian/serving.py
```

## Hardware inputs used for normalization

| GPU basis | Count | VRAM | Memory bandwidth | FP16/BF16 peak used |
| --- | ---: | ---: | ---: | ---: |
| Radeon RX 7700 | 1 | 16 GB | 624 GB/s | 50.4 TFLOPS FP16 matrix |
| RTX PRO 6000 Blackwell Server | 8 | 768 GB aggregate | 12,776 GB/s aggregate | 8,000 TFLOPS FP16/BF16 aggregate |
| A100 80GB SXM | 1 | 80 GB | 2,039 GB/s | 312 TFLOPS dense BF16 |

AMD advertises 16 GB, 624 GB/s, 25.2 TFLOPS FP16 vector, and 50.4 TFLOPS
FP16 matrix for the RX 7700. AMD does not publish a BF16 peak on that product
page, so the requested compute normalization uses the advertised FP16 matrix
number as an explicit proxy. NVIDIA advertises 96 GB, 1,597 GB/s, and 1 PFLOP
FP16/BF16 per RTX PRO 6000. NVIDIA's A100 specification lists 80 GB, 2,039 GB/s,
and 312 TFLOPS dense BF16; the larger 624 TFLOPS figure requires sparsity and is
not used here.

Sources: [AMD RX 7700 specifications](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700.html),
[NVIDIA RTX PRO 6000 Server specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/),
and [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/).

## External public results

These rows are not apples-to-apples results. The models, precisions, request
shapes, software, GPU architectures, and GPU counts differ. Neither SGLang nor
MAX/Mojo ran this workload locally: both failed to execute on `gfx1101`. Their
published datacenter-GPU results are external context only.

| System | GPU | Model/workload | Published output tok/s | RX-7700-equivalent theoretical range |
| --- | --- | --- | ---: | ---: |
| Severian | RX 7700 | Qwen2.5-3B BF16, 256-in/32-out, 80 requests, c16 | **19.747 measured** | **19.747 measured, not translated** |
| SGLang | 8× RTX PRO 6000 | Qwen3.5-122B-A10B FP8, BF16 KV, 500×200-in/200-out burst | 1,985 | 12.506–96.951 tok/s theoretical |
| MAX/Mojo | 1× A100 80GB SXM | Llama 3.1 8B BF16, ShareGPTv3, 500-prompt burst | 3,860 | 623.538–1,181.285 tok/s theoretical |

The SGLang number is a reproducible community result posted in the SGLang
project issue tracker, not an official vendor performance claim. It reports a
three-run warm mean of 1,985 ± 11 output tok/s. The MAX value is Modular's own
published claim. Modular's detailed methodology identifies one A100-80GB SXM,
BF16, Llama 3.1 8B, 500 prompts, and five measured runs after five warmups.

References:

- [SGLang `bench_serving` guide and metric definitions](https://docs.sglang.ai/developer_guide/bench_serving)
- [SGLang public 8× RTX PRO 6000 result](https://github.com/sgl-project/sglang/issues/19603)
- [MAX 3,860 output tok/s announcement](https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform)
- [MAX A100 benchmark methodology](https://www.modular.com/blog/max-gpu-state-of-the-art-throughput-on-a-new-genai-platform)
- [Current MAX benchmark documentation](https://docs.modular.com/serve/benchmark/)

## Mechanical scaling context

The requested calculations use output throughput only:

```text
throughput_per_bandwidth = output_tok_s / aggregate_bandwidth_GBps
throughput_per_compute   = output_tok_s / aggregate_peak_TFLOPS
```

| System | tok/s per GB/s | tok/s per peak TFLOP |
| --- | ---: | ---: |
| Severian | 0.031646 | 0.391803 |
| SGLang public | 0.155369 | 0.248125 |
| MAX public | 1.893085 | 12.371795 |

Translated independently to the RX 7700 hardware inputs:

| External result | Pure bandwidth scaling | Pure peak-compute scaling | Severian / bandwidth-scaled | Severian / compute-scaled |
| --- | ---: | ---: | ---: | ---: |
| SGLang | 96.951 tok/s | 12.506 tok/s | 0.204× | 1.579× |
| MAX/Mojo | 1,181.285 tok/s | 623.538 tok/s | 0.0167× | 0.0317× |

Thus Severian's measured 19.747 tok/s lies inside the SGLang mechanical range
of 12.506–96.951 tok/s and below the MAX range of 623.538–1,181.285 tok/s.
These endpoints are separate scaling thought experiments, not predictions; they
must not be averaged. In particular, the radically different models, FP8 versus
BF16 weights, speculative decoding, request lengths, batching behavior, and
multi-GPU communication make the ratios unsuitable as benchmark rankings.
