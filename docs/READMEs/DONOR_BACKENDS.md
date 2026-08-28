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

## Region artifact routing

`CompileHandler` returns a backend-neutral `CompiledRegionArtifact`. Host and
SIMD tensor regions produce `CpuMlir`; GPU tensor regions produce a
`GpuKernelBundle` containing the complete `FusionGraph`, its selected
`FusionPlan`, target architecture, and typed region signature. The compiler
registry verifies only the CPU variant as an MLIR artifact. A GPU bundle never
enters the CPU tensor emitter or MLIR artifact verifier.

The ordinary host module contains a small `__sev_artifact_N` wrapper calling
`__sev_gpu_launch_N`. `RoutedProgram` retains the corresponding GPU bundles
beside that host MLIR so later Triton compilation and runtime packaging do not
have to reconstruct or recover the execution graph from host code.

## Ownership boundary

`severian-fusion` owns the complete producer/consumer graph, shape and cost
facts, alias/effect facts, fusion decisions, and the mapping from every graph
node to its selected region. It adapts the priority rule
`time_unfused - time_fused` and the parameter/shared-memory/reduction budgets
from XLA:GPU without depending on `HloInstruction`.

Graph tensor metadata is structural data, not part of an opcode or symbol.
Each node distinguishes ranked (including rank zero) from unranked tensors,
known from dynamic dimensions, exact element kind and bit width, dense,
strided, or runtime layout, operand roles, storage aliases, and mutation.
Runtime dimensions and strides enter compilation through a
`KernelSpecialization { shapes, strides, target }`. Specialization selects a
kernel instance for a fusion region; it never manufactures dtype- or
rank-specific operations such as `matmul_f16` or `matmul_rank4`.

`severian-triton` owns both `FusionRegion`-to-TTIR construction and ABI
versioning. Its public compiler boundary accepts the graph, selected region,
runtime specialization, and target options—never caller-authored TTIR text.
Only the private native bridge request contains TTIR bytes, produced inside
`severian-triton`, alongside pointer-stable graph views. The ABI is
intentionally one-way: Triton receives TTIR and returns an optimized kernel
artifact. It does not return a replacement execution graph and cannot redefine
Severian types or semantics.

TTIR emission is selected by structural operation class. Severian owns tensor
semantics, concrete indexing, bounds masks, and rank-generic contraction
dimensions. Triton owns thread layouts, shared-memory placement, warp and stage
scheduling, and target lowering. Generated modules are parser-tested with
`triton-opt` from the exact revision pinned in `compiler/donors.toml`.

The initial pass-order transcription is represented as Rust data in
`pass_pipeline`. The native C++ bridge maps the donor passes to Triton's C++
constructors. Python is not part of the production compilation boundary.

## GPU runtime and specialization cache

`severian-runtime::gpu` owns device discovery, buffers and transfers, kernel
loading, ABI argument packing, grid calculation, launch events, and dependency
scheduling across fusion regions. CUDA, HIP, remote, and test executors all
implement the same `GpuDriver` contract. Triton implements `GpuCompiler`; it
does not own runtime resources or scheduling.

Fusion-region dependencies are derived from values crossing `FusionPlan`
region boundaries and emitted in stable topological order. Launches receive
the precise predecessor events, allowing a driver to schedule asynchronously
without imposing a global synchronization after every region.

Compiled kernels have a memory cache and an optional persistent cache. A
deterministic key covers the complete graph and selected region, concrete
shape and stride specialization, every node's element kind and bit width,
target architecture, the pinned Triton donor revision, and all compiler
options. Cached records include binary format, entry point, code, grid policy,
block shape, warps, CTAs, and shared-memory requirements. Rank and dtype remain
ordinary graph/ABI data; cache specialization never creates rank- or
dtype-suffixed source functions.

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
