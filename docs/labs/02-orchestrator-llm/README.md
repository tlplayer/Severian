# Three-replica tiny-LLM orchestrator lab

This lab joins the compiler, tensor runtime, device inventory, CDI, OCI,
networking, health probes, reconciliation, and service routing in one runnable
Severian workload.

```text
client -> 127.0.0.1:8080 -> orchestrator gateway
                                  |
                    +-------------+-------------+
                    |             |             |
                 worker-0      worker-1      worker-2
                 :18181        :18182        :18183
                    |             |             |
                 OCI/CDI       OCI/CDI       OCI/CDI
                    \_____________|_____________/
                                  |
                         tensor CompileType
```

The control plane schedules exactly three replicas, gives each a distinct
container identity and port, probes `/live` and `/ready`, routes only to ready
workers, exposes aggregate metrics, and emits restart actions for failed
replicas. Accelerator requests are generic orchestrator device claims. A
selected AMD device becomes a CDI identity such as `amd.com/gpu=0`; OCI records
that identity without exposing `/dev/kfd` or render-node details to the app.

## Run

From this directory:

```bash
sev run
```

`tiny-llm-cluster` builds its worker, starts three OCI containers when the host
runtime accepts the bundles, and otherwise starts the same three workers as
native development processes. Leave it running and use a second terminal:

```bash
sev run --bin tiny-llm-client -- "GET /status"
sev run --bin tiny-llm-client -- "GET /metrics"
sev run --bin tiny-llm-client -- "GET /info"
sev run --bin tiny-llm-client -- generate "2 + 2 ="
sev run --bin tiny-llm-client -- generate "The capital of France is"
```

The endpoint-shaped development protocol is deliberately tiny:

```text
POST /generate\nPROMPT
GET /live
GET /ready
GET /metrics
GET /info
GET /status            # gateway aggregate
```

The gateway listens on `127.0.0.1:8080`. Direct worker ports are `18181`
through `18183`; container processes listen on `19181` through `19183` and are
reached through the OCI port forwards.

After stopping the gateway, remove the development containers with:

```bash
for replica in 0 1 2; do
  crun --root /tmp/severian-crun kill "tiny-llm-$replica" TERM || true
  crun --root /tmp/severian-crun delete --force "tiny-llm-$replica" || true
done
```

## Model boundary

[`model/model.toml`](model/model.toml) declares
`HuggingFaceTB/SmolLM2-135M-Instruct`, BF16, greedy decoding, batch size one,
and at most 32 generated tokens. The lab no longer contains a smoke classifier or checked-in substitute
weights. `ai.model.load` dispatches to the SmolLM2 provider, pins revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, and acquires `config.json`, the
official tokenizer, and `model.safetensors` into `target/models`.

Readiness requires the complete 269,060,552-byte safetensor artifact. A worker
must not advertise readiness or fabricated generation while decoder lowering
is unavailable. The compiler component cache contains the official StableHLO
legalization library and LLVM 21's ROCm runtime wrapper; generation becomes
available only when the complete Llama graph lowers and executes through that
route.

## Validate

```bash
sev test
sev test ../../../library/system/orchestrator
sev test ../../../library/system/oci
```

The orchestrator tests cover three distinct GPU allocations, portable native
fallback, readiness-gated round robin, failure thresholds, and restart
generation. OCI tests cover process arguments, CPU/memory limits, and CDI
annotations.
