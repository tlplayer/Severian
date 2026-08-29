# Artifact routing

Stable IDs: `compiler.artifact.*`.

`CompiledRegionArtifact` is backend-neutral and has three structurally distinct
results:

- `CpuMlir` contains verifier-valid host compute MLIR.
- `GpuKernel` contains the Severian fusion graph, schedule, target, value-node
  mapping, signatures, and architecture needed for GPU lowering.
- `TensorJit` retains a graph only when rank remains unresolved at build time.

GPU regions never masquerade as CPU tensor functions. The ordinary host module
receives launcher calls. Verification wraps raw artifacts before composition,
so an artifact's target, entry signature, and route cannot silently change.

Relationship: `ArtifactId` is cache/composition identity; it is not an
operation ID, dtype, rank, or source function specialization.
