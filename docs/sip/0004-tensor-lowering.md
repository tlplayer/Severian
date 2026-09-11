# SIP: Staged numeric tensor lowering for SIMD and GPU

**Dynamic contracts at the boundary; concrete arithmetic inside the kernel.** Unknown dimensions do not require dynamically typed elements—or necessarily JIT compilation. MLIR can represent tensors with runtime dimensions while retaining rank and element type. ([MLIR][1])

[Download the SIP](sandbox:/mnt/data/SIP-staged-numeric-tensor-lowering.md)

**Status:** Proposed. Covers types, lowering, specialization, inference compatibility, migration, and acceptance tests. Source-reviewed; not implementation-tested.

## 1. Build on the existing pipeline

Severian already has `PrimitiveRepresentation`, compiled regions, and runtime-specialized CPU/GPU entry points. The ordinary GPU path currently rejects unresolved rank, while the specialized entry point can refine the region before GPU emission. Extend that mechanism rather than introducing another JIT provider. ([GitHub][2])

The current GPU emitter handles equal-contract elementwise operations. It also reconstructs scalar semantics from type spelling: non-floating division uses signed division, and ReLU uses signed comparison. **Fix that before extending GPU coverage.** ([GitHub][3])

The architectural change is:

> Preserve resolved numeric operators and tensor constraints through the pipeline. SIMD and GPU lowering choose execution schedules—not separate interpretations of arithmetic.

## 2. Represent different kinds of unknowns separately

| Unknown at compilation       | Required representation             | Execution strategy                                                 |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| Scalar value                 | Typed runtime argument              | AOT; pass the value.                                               |
| Tensor dimensions            | Symbolic dimensions and constraints | AOT loops/masks when supported; optional specialization.           |
| Tensor rank                  | Explicit unknown-rank contract      | Generic descriptor kernel, or bind rank and dispatch/JIT.          |
| Element dtype                | Runtime dtype witness               | Resolve dtype and promotion, then dispatch/JIT.                    |
| Strides/alignment            | Runtime layout metadata             | Generic strided kernel or guarded specialization.                  |
| Data-dependent output length | Runtime cardinality                 | Dynamic loops, count-then-allocate, or capacity plus valid length. |

**Unknown rank is not rank zero. Unknown dtype is not `f64`, `bf16`, or `Any`.**

For example:

```text
Tensor[f32], shape = [batch, sequence, 1024]
```

There is enough information to compile indexing, arithmetic, and runtime loop bounds. Specializing every sequence length would be a performance choice—not a correctness requirement.

By contrast:

```text
Tensor[numeric], dtype and rank supplied at runtime
```

needs a representation that preserves those unresolved facts until execution. The runtime can select an existing implementation or compile a specialization once the required facts become known.

**JIT does not require a dynamically typed language. It requires retaining the program and its unresolved constraints until specialization is possible.**

## 3. Define `Tensor[numeric]` without boxing every element

Proposed meanings:

| Source type       | Meaning                                                                        |
| ----------------- | ------------------------------------------------------------------------------ |
| `Tensor[f32]`     | Homogeneous storage with a known representation.                               |
| `Tensor[T]`       | Homogeneous storage parameterized by `T`.                                      |
| `Tensor[numeric]` | Homogeneous storage whose concrete numeric dtype is witnessed at runtime.      |
| `list[numeric]`   | Potentially heterogeneous numeric values; not automatically dense GPU storage. |

Conceptually:

```text
Tensor[numeric] = exists D in supported numeric representations: Tensor[D]
```

Not:

```text
Tensor[numeric] = array of individually tagged numeric objects
```

A proposed source contract could be:

```sev
def add(a: Tensor[numeric], b: Tensor[numeric]) -> Tensor[numeric]:
    return a + b
```

At a compiled-region boundary:

```text
Read operand dtype witnesses
→ resolve existing promotion/conversion rules
→ bind concrete operand/result representations
→ select or compile a kernel
→ execute unboxed arithmetic
```

The dtype dispatch happens **once per region**, not once per element.

This must not silently convert arbitrary-precision integers or other managed numeric objects into machine floats. Packing heterogeneous values requires an established promotion policy or an explicit target dtype. Values that cannot satisfy the requested representation stay on an appropriate execution path or produce a conversion error.

### Keep one representation authority

Runtime dtype metadata should identify the existing `PrimitiveRepresentation`, not introduce another independently maintained dtype table.

Preserve:

```text
representation + shape constraints + strides + offset
+ storage extent + ownership/alias facts + device
+ precision policy + execution readiness
```

For quantized storage, also preserve packing, grouping, scales, zero points, and accumulator requirements.

**Runtime specialization adds information. It must not erase the original contract.**

## 4. Share arithmetic semantics; separate CPU and GPU scheduling

Proposed pipeline:

```text
Severian source                  Imported PyTorch graph
       |                          semantic normalization
       +--------------------------------+
                                        |
                 Universal numeric/tensor/effect contracts
                                        |
                     HIR/MIR regions + constraints
                                        |
                    AOT or bind → guard → specialize
                                        |
                  Shared operations and scalar bodies
                         /                       \
                  CPU schedule                GPU schedule
                loops + vector              tiled GPU operations
                    LLVM                    ROCDL / NVVM
```

MLIR supplies vector and GPU operations, but the compiler still owns legality, scheduling, bufferization, and runtime integration. This is an architecture, not a claim that one pass sequence automatically produces performance. ([MLIR][4])

The shared operation must retain its resolved declaration, operand/result representations, conversions, indexing rules, reduction axes, effects, and source provenance. Do not reduce it to `"divide"` plus `"i8"` and reconstruct semantics downstream.

**CPU SIMD:** choose vector width from target capabilities; handle tails; preserve stride and alias constraints; retain scalar execution when vectorization is illegal or unprofitable.

**GPU:** add broadcasting, reductions, contractions, normalization, attention, gather/scatter, and required convolution/layout operations to the existing path. Use actual device storage, launch bounds, events, and ownership tracking.

`with gpu` should mean strict tensor-compute placement. Unsupported execution must produce a diagnostic—not silently run the tensor operations on CPU. Host scheduling and shape bookkeeping remain allowed.

## 5. Specialize only what improves or enables execution

The default policy should keep **batch, sequence length, cache position, and valid lengths dynamic** where the lowering supports them.

Specialize dtype, required rank, device capabilities, and useful layout properties. Specialize model constants only when their constancy is established.

### Cache identity

Include:

```text
region identity + transitive package content hashes
+ compiler/backend/ABI compatibility
+ device features
+ concrete representations
+ semantic and precision policy
+ facts actually assumed by generated code
```

Do not include buffer addresses, ordinary input values, or changing dimensions unless the generated code genuinely specializes them. Weight contents belong in the key only when deliberately constant-folded or specialized.

A guard miss on a legal input means **select another variant or compile one**. It is not a type error. Run guards before mutation or RNG consumption so retries cannot replay effects.

Start with a configurable budget—for example, eight optimized variants per region/target/policy namespace, plus a global code-memory limit. That is an initial policy, not a measured optimum. At the limit, use a compatible generic implementation or return an explicit budget/capability error.

Allow packages to contain precompiled variants and retained region IR. AOT-only deployments should not need to embed a compiler.

## 6. Match PyTorch semantics at import—not throughout the language

Support two entry paths:

```text
Native Severian model + validated weights/configuration
PyTorch exported program + normalization/import
```

Both lower into the same numeric/tensor contracts.

For exported programs, use graph signatures, symbolic constraints, and an explicit decomposition policy. `run_decompositions()` can normalize exported operations and represent mutations functionally; do not assume every incoming export already has the required form. ([PyTorch Documentation][5])

Compatibility must cover promotion, broadcasting, indexing, reduction behavior, casts, layouts, and state updates—not merely operator names.

For example, Severian’s integer `/` is defined as truncating toward zero. Imported PyTorch true division must retain its own resolved semantics and casts rather than inherit native integer division because both use `/`. PyTorch also distinguishes scalar and tensor categories during promotion. ([GitHub][6])

### Separate semantic compatibility from numerical optimization

| Policy                  | Requirement                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Compatible, default** | Preserve pinned operation dtypes, casts, accumulation requirements, errors, and agreed tolerances.                                               |
| **Fast, explicit**      | Permit selected approximation, reassociation, reduced-precision accumulation, TF32, or quantization. Record choices in artifacts and cache keys. |

Storage dtype, compute dtype, and accumulator dtype are separate.

Keep BF16/F16 weights in their storage format. Widen accumulation where the operation requires it. Do not convert everything to F32 or BF16 to simplify lowering.

**Fusion may remove a memory write; it may not silently remove an observable rounding step.** Qwen3’s reference attention computes softmax in F32 and casts back to the query dtype before the next matmul. A compatible lowering must preserve that boundary unless its precision policy permits a deviation. ([GitHub][7])

Compatibility should use operation- and dtype-specific tolerances, not a universal promise of bitwise identity. PyTorch itself documents numerical differences across implementations and execution arrangements. ([PyTorch Documentation][8])

## 7. Use models as acceptance tests

### Qwen: prefill and decode are separate workloads

Start with one pinned dense decoder checkpoint. Its fixture must cover embedding, normalization, RoPE, grouped-query attention, masks, gated MLP, and KV-cache behavior—not just matrix multiplication. Qwen3’s implementation provides a reference for those operations. ([GitHub][7])

The proposed execution contract is:

```text
Prefill: dynamic prompt length; reusable compiled regions.
Decode: runtime cache position and valid length.
Weights and KV cache: device-resident.
Within warmed shape constraints: no per-token recompilation.
```

Validate intermediate tensors and logits before generated text. Add MoE and multimodal variants separately; data-dependent routing must not become a constant inferred from the compilation sample.

### Muse: pin the architecture before claiming support

Here, **Muse means the masked image-token transformer family**. The original Muse uses text conditioning and masked-token generation with image tokenization/decoding—not a diffusion denoising loop. The exact checkpoint remains unspecified. ([Muse Model][9])

A pinned fixture should exercise its text encoder, masked transformer, sampling/remasking schedule, and image-code decoder, including the decoder’s required convolution/layout operations.

Compare per-iteration logits, supplied random draws, token decisions, and decoded outputs. A common seed alone is not a sufficient cross-implementation RNG contract.

These workloads test different requirements: Qwen stresses recurrent cache use; Muse stresses iterative tensor execution and image decoding.

## 8. Implementation stages and deletion gates

| Stage                              | Implementation                                                              | Required evidence                                                            |
| ---------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **0. Lock semantics**              | Add signedness, dtype-loss, shape, and error regressions.                   | GPU unsigned division and ReLU preserve `u8(200)` semantics.                 |
| **1. Preserve unknowns**           | Add runtime dtype witnesses and retain constraints.                         | Runtime inputs survive lowering without becoming `f64`, `Any`, or rank zero. |
| **2. Complete SIMD path**          | Dynamic loops, vector scheduling, tails, strides.                           | Correct results and verified vector instructions.                            |
| **3. Connect GPU specialization**  | Reuse existing specialization entry points; add guards and bounded caching. | Unknown metadata reaches actual device execution.                            |
| **4. Extend operation coverage**   | Reductions, matmul, attention, gather, required convolution/layouts.        | Per-operation numerical and ownership tests pass.                            |
| **5. Validate models and migrate** | Pinned Qwen/Muse fixtures; remove superseded flows.                         | Model outputs, residency, cache reuse, and profiling gates pass.             |

The public tensor source still exposes `ranked(...) -> Tensor[f64]` and `values[T](...) -> list[float]`. Migrate to ordinary constructors and dtype-preserving extraction before deleting those interfaces. **Keep ranked internal MLIR types; remove rank-specific public workarounds.** ([GitHub][10])

Also remove backend arithmetic decisions based on strings once shared semantic lowering replaces them. Do not revive the retired C/Triton JIT route.

The acceptance suite must build an entry once and supply dtype/shape/storage metadata externally at runtime. Otherwise, constant inputs can conceal missing dynamic support.

Measure cold startup, compilation, warmed latency, prefill throughput, decode time/token, host/device memory, transfers, specialization count, and actual GPU execution. Compare identical model, dtype, precision, and synchronization settings.

**The completion criterion is not “everything is dynamic.” It is that unknown information survives until needed, then disappears from the kernel’s arithmetic cost without changing the program’s meaning.**

[1]: https://mlir.llvm.org/docs/Dialects/Builtin/ "https://mlir.llvm.org/docs/Dialects/Builtin/"
[2]: https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/compiler/src/lib.rs "https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/compiler/src/lib.rs"
[3]: https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/compiler/src/gpu.rs "https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/compiler/src/gpu.rs"
[4]: https://mlir.llvm.org/docs/Dialects/Vector/ "https://mlir.llvm.org/docs/Dialects/Vector/"
[5]: https://docs.pytorch.org/docs/2.8/export.html "https://docs.pytorch.org/docs/2.8/export.html"
[6]: https://raw.githubusercontent.com/tlplayer/Severian/main/sev_compiler/universal/primitive/numeric/operators.sev "https://raw.githubusercontent.com/tlplayer/Severian/main/sev_compiler/universal/primitive/numeric/operators.sev"
[7]: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py"
[8]: https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html "https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html"
[9]: https://muse-model.github.io/ "https://muse-model.github.io/"
[10]: https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/src/lib.sev "https://raw.githubusercontent.com/tlplayer/Severian/main/library/compute/tensor/src/lib.sev"
**The SIP should treat memory capacity as an execution constraint, not require the model to remain resident.** The model program stays unchanged; storage encoding, residency, tile size, and prefetching become configurable.

[Updated SIP — revision 2](sandbox:/mnt/data/SIP-staged-numeric-tensor-lowering-v2.md)

This replaces the earlier device-resident weights/KV requirement with a resident **baseline profile**, alongside streaming profiles. 

## 1. Separate compression, precision, and streaming

These should be independent controls:

| Mechanism              | What changes                                 | Required guarantee                                                                        |
| ---------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Lossless compression   | Stored bytes and decoding work               | Recover the original tensor values exactly.                                               |
| Low-bit representation | Representable values or approximation policy | Preserve a native low-bit model’s semantics, or validate an explicitly quantized variant. |
| Streaming/offloading   | Location and movement of data                | Execute within memory limits without changing the selected model.                         |

“One-bit” should be one supported encoding—not the organizing abstraction. BitNet b1.58, for example, uses ternary weights and model-specific quantization semantics. Its native training approach does not establish a lossless conversion of arbitrary Qwen or Muse weights into one bit. ([arXiv][1])

Nor should we assume compression always sacrifices speed. Packed kernels can reduce bandwidth and computation enough to improve both speed and memory use; Bitnet.cpp demonstrates this for ternary inference. Severian should measure which tradeoff applies to each plan. ([arXiv][2])

## 2. A tensor must not imply an allocation containing every element

Extend the existing tensor contract rather than introducing separate “streaming tensor” arithmetic:

```text
Tensor
    logical numeric type and shape
    storage encoding and decoding rules
    backing resource and block index
    residency, ownership, and readiness
```

A tensor can therefore represent data backed by a checkpoint without allocating its full expanded representation.

Keep these facts separate:

```text
stored representation   → packed codes, scales, optional compression
compute representation  → the operand representation used by a kernel
accumulator              → the representation required by the operation
residency                → disk, host memory, or device memory
```

MLIR already distinguishes expressed values from quantized storage values. That provides part of the representation model; it does not supply the streaming runtime. ([MLIR][3])

**Dispatch at the region or tile boundary, not per element.** Runtime encoding metadata selects a supported kernel or bounded decoder. Moving a block between memory tiers should not change its logical type or trigger recompilation.

## 3. Lower operations into bounded working sets

The execution path should support:

```text
checkpoint → host staging → packed device tile → decode/compute → release
```

Prefer fused unpack/dequantization and computation where supported. Otherwise, expand only the tile being consumed—not the whole tensor or model.

The runtime should acquire a tile, wait for readiness, execute its consumers, and release its storage only after their completion events. Immutable weights can be discarded from the device without copying them back. Modified KV pages need separate writeback handling.

Prefetching should be optional and bounded. Two buffers may allow the next block to transfer while the current block executes; a smaller budget may require serialization. Existing group-offloading implementations illustrate both transfer overlap and its additional host-memory costs. ([Hugging Face][4])

The planner must account for:

```text
peak memory =
    resident weights
  + live activations and KV/state
  + transfer buffers
  + decoded tiles
  + operator workspaces
  + runtime/compiler overhead
```

**A compressed checkpoint fitting in memory does not prove execution fits.** The same accounting must apply during downloading, loading, repacking, and compilation.

When an entire layer exceeds the budget, use operator tiling where legal. Otherwise report the operation and minimum supported working set. Do not silently shorten context, reduce image resolution, or switch to CPU.

For strict GPU execution, model arithmetic and tensor dequantization stay on GPU. Host I/O and scheduling remain allowed.

## 4. Make the harness select an execution plan

The library workflow should be:

```text
inspect metadata
→ resolve/pull an immutable model revision
→ validate supported operations and encodings
→ plan against memory and workload constraints
→ execute
→ validate and report measurements
```

Revision-pinned downloads and version-aware caching are already supported by the Hugging Face acquisition interface. The harness should preserve that identity through derived encodings and execution reports. ([Hugging Face][5])

Do not overwrite the original checkpoint when producing another representation. Store derived artifacts separately with their parent hashes, encoding, preparation parameters, and quality results.

Proposed harness settings; the budgets are illustrative:

```toml
[model]
repository = "Qwen/Qwen2.5-3B-Instruct"
revision = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"
variant = "native"
allow_requantization = false

[execution]
compute = "gpu"
fallback = "error"
objective = "latency"

[memory]
device_budget = "4 GiB"
host_budget = "8 GiB"
disk_cache_budget = "20 GiB"

[streaming]
weights = "auto"
kv_cache = "device"
max_staged_block = "64 MiB"
max_inflight_blocks = 2
network_during_execution = false
```

Here, `auto` chooses residency and scheduling—not a different numerical model.

A native low-bit checkpoint can use its packed representation directly. A quantized Qwen variant must be selected and validated explicitly. Weight streaming and KV paging remain separate settings.

The default should pull the selected artifact once, then stream from local storage. Network-backed execution can be another policy, rather than accidentally downloading weights repeatedly during generation.

### Capacity and latency need separate acceptance criteria

The planner should expose the transfer cost before execution. For example, assuming batch-one decode transfers **4 GB per token** across a link sustaining **16 GB/s**:

$$
t_{\text{transfer}}\geq \frac{4}{16}=0.25\text{ seconds/token}
$$

That schedule cannot exceed four tokens per second, even before accounting for other bottlenecks. These are illustrative assumptions, not measurements.

Caching, compression, batching, and overlap can change the result. This is why the objective must distinguish interactive latency from batch throughput; FlexGen’s resource-constrained inference work makes that distinction relevant. ([arXiv][6])

## 5. Test that streaming actually happens

The acceptance harness should force the condition it claims to support:

| Test                                | Required evidence                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| Model exceeds device budget         | Correct output under an enforced cap, with measured transfers and peak allocation.     |
| Expanded weights exceed host budget | Loading and preparation succeed without constructing a full dense model.               |
| A tensor exceeds the tile pool      | Tiled execution succeeds, or reports the minimum working set.                          |
| Packed execution                    | No hidden whole-model expansion; encoding boundaries and scales are correct.           |
| Asynchronous execution              | Delayed transfers and cancellation cannot cause premature buffer reuse.                |
| Changing runtime conditions         | No compilation per token, tile address, or prefetch operation; no hidden CPU fallback. |

Quantization needs two checks: execution against the **same encoded model’s reference**, then quality against the **original checkpoint**. Passing the first does not establish the second.

**Implementation order:** bounded unquantized streaming first; one packed format next; then prefetching and plan selection; finally KV paging and model-specific workload coverage.

The target becomes: **run the same model on supported machines by changing its execution plan, within declared memory and quality limits—not by changing its meaning to make it fit.**

[1]: https://arxiv.org/abs/2504.12285 "BitNet b1.58 2B4T Technical Report"
[2]: https://arxiv.org/abs/2502.11880 "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs"
[3]: https://mlir.llvm.org/docs/Quantization/ "Quantization - MLIR"
[4]: https://huggingface.co/docs/diffusers/optimization/memory "Reduce memory usage · Hugging Face"
[5]: https://huggingface.co/docs/huggingface_hub/guides/download "Download files from the Hub · Hugging Face"
[6]: https://arxiv.org/abs/2303.06865 "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"
