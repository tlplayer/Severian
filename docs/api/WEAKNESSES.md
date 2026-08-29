# API weakness ledger

This file is deliberately adversarial: it lists places where the advertised
architecture is stronger than the executable implementation. Closing a row
requires evidence and an API status update, not merely deleting the prose.

| ID | Surface | Current weakness | Evidence | Exit criterion |
| --- | --- | --- | --- | --- |
| `W-API-001` | API reference generation | Native `sev api list/show/check/diff` exists, but `docs/reference/` and semantic IDE metadata are not generated from it yet. | `compiler/boundaries/driver/src/api.rs` validates and queries the records without a renderer. | Reference pages and semantic editor metadata are reproducibly generated from the checked catalogue. |
| `W-GEN-001` | Dependency interfaces | Downstream self-hosted packages cannot consistently consume dependency enum variants/patterns, forcing record views at package boundaries. | The `.sev` kernel-to-MLIR slice uses `operation_record` and `index_operation_record`. | Dependency interfaces preserve and resolve enum constructors and generic bodies without semantic special cases. |
| `W-MLIR-001` | Self-hosted final lowering | `.sev test` may emit aggregate type aliases before aliases they reference. | A generated `!sev_class_20` referred to later `!sev_class_33`/`!sev_class_34`. | Aggregate aliases are topologically ordered or emitted as legal recursive identified types. |
| `W-GPU-001` | GPU compiler | The `.sev` object pipeline currently covers a blocked elementwise slice, not reductions, contractions, or target machine code. | `sev_compiler/transforms/kernel` and `gpu_mlir` records. | Reduction and rank-generic matmul lower through verified GPU MLIR to ROCDL/NVVM. |
| `W-GPU-002` | Driver integration | The ordinary driver does not yet produce `.sev` `LogicalKernel` objects from every `FusionRegion`. | The tensor example constructs the initial kernel contract through a fixture. | Driver executes `FusionRegion → LogicalKernel → ScheduledKernel → verified MLIR`. |
| `W-IDE-001` | Editor semantics | VS Code support is TextMate/CLI based; it has no compiler-backed semantic tokens, completion, or navigation. | `editors/vscode` contains no language-server client. | LSP uses compiler symbol/type/ownership data and API IDs. |
| `W-TENSOR-001` | Tensor surface | Structural operation identities exist, but backend coverage is not exhaustive across every element kind, rank, layout, and effect. | `TensorOp::ALL` exceeds current direct GPU lowering coverage. | Generated conformance matrix passes all legal combinations and rejects illegal ones at legalization. |
| `W-TEST-001` | Behavioral symmetry | Python/Rust symmetry currently covers scalar operators and ordinary generics, not all 498 queryable records. | `api/SYMMETRY.md` lists two passing cases and the missing groups. | Every implemented API group has a behavioral or compile-outcome oracle; partial groups expose failing cases at their precise boundary. |

## How records expose weakness

Every `partial`, `experimental`, or `unavailable` record must contain a
non-empty `limitations` list. Backend-specific gaps belong in
`backend_status`; they must not be hidden by marking a structural operation
globally implemented.

Symmetry cases may name a weakness when Severian cannot yet match its Python
or Rust reference. That expected gap remains a failing/blocked capability in
the report; it is never silently counted as a pass.
