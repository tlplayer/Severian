# Donor GPU backends

Severian treats OpenXLA and Triton as code donors, not as source-language or
IR authorities. HLO, StableHLO, the Python Triton frontend, and PJRT framework
glue do not become part of the Severian compiler model.

The intended pipeline is:

```text
Severian Tensor IR
  -> severian-fusion (XLA-derived eligibility, budgets, and priority policy)
  -> FusionRegion
  -> Severian FusionRegion-to-TTIR lowering
  -> severian-triton ABI
  -> Triton TTIR/TritonGPU/target pass pipeline
  -> HSACO or PTX/CUBIN
  -> Severian runtime
```

CPU compilation remains on the existing Linalg/Vector/LLVM route.

## Ownership boundary

`severian-fusion` owns the complete producer/consumer graph, shape and cost
facts, alias/effect facts, fusion decisions, and the mapping from every graph
node to its selected region. It adapts the priority rule
`time_unfused - time_fused` and the parameter/shared-memory/reduction budgets
from XLA:GPU without depending on `HloInstruction`.

`severian-triton` owns ABI versioning and converts a selected region plus TTIR
into pointer-stable C views for a native donor bridge. The ABI is intentionally
one-way: Triton receives TTIR and returns an optimized kernel artifact. It does
not return a replacement execution graph and cannot redefine Severian types or
semantics.

The initial pass-order transcription is represented as Rust data in
`pass_pipeline`. The native C++ bridge will map those names to Triton's pass
constructors, replacing the orchestration currently performed by AMD and
NVIDIA `backend/compiler.py`.

## Reproducibility and licenses

Exact reviewed donor revisions and relevant source paths live in
`compiler/donors.toml`. OpenXLA is Apache-2.0 and Triton is MIT licensed. Any
copied or translated implementation must retain the applicable notices; the
Severian implementations identify their donor provenance in module comments.

Neither donor checkout belongs inside the Severian repository. The expected
development layout is:

```text
Documents/
  Severian/
  triton/
  xla/
```
