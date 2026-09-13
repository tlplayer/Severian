# Ownership path audit

Audit date: 2026-09-12. This records active paths separately from preserved
compatibility implementations. The migration is not complete.

| Path | Finding | Action/status |
| --- | --- | --- |
| Rust driver `compile_mir` | Scalar programs could select direct LIR-to-C code generation and skip MLIR. | Driver now always composes MLIR. C backend implementation is preserved. |
| Rust MLIR backend | One-shot bufferization allocated buffers without the ownership deallocation pipeline. | Runs the ownership library's `mlir.pipeline` before LLVM conversion. |
| Source MLIR backend | Used standard buffer deallocation, with pass ordering copied into the backend. | Reads the same ownership library pipeline. Native cache inputs include that file; checking and MLIR-only emission do not require hosted C provider files. |
| Test function pruning | Name-based preservation omitted registered payload destruction callbacks. | Uses the type contract’s registered storage-glue identities, including payload cleanup. |
| Source semantic parameter effects and loans | Policy lived in semantic helpers, apart from the ownership package. | Implementations now live in ownership; old module paths forward to them. |
| Source MIR availability and stack escapes | Lowering authority lived in MIR. | Plan and availability implementations now live in ownership; MIR forwards to them. |
| `core.memory` typed buffers | `memref.alloc` already preserved ownership in MLIR. | Added source zeroed allocation and resize equivalents; C providers retained. |
| `core.storage` typed buffers | Source checked allocation and consuming drop existed. | Added initialized independent copy using `memref.copy`; no pointer conversion. |
| Runtime and POSIX allocations | Physical allocations routed through `core.memory`, except regex and network helpers. | Regex/network helpers now use that provider too; foreign OS destructors remain foreign adapters. |
| Explicit release checking | A raw release helper lacked a consuming parameter contract. | `raw_free` now takes `move pointer[T]`; ownership rejects releasing a view with `E000303` and the offending span. |
| Explicit `pointer[T]` allocation/free | Active prelude imports `core.memory/src/raw_intrinsics.sev`; allocation/free cross its explicit hosted adapter. Raw pointers lose MLIR owner identity. | Preserved unsafe compatibility path. Requires owner/provenance representation before replacement, especially address-of and pointer casts. |
| `universal/primitive/hosted_memory.sev` | Old duplicated declarations; not an active prelude provider. | Preserved; active source imports the canonical memory library already. |
| Bootstrap strings/lists/boxed values and enum payloads | Values become opaque pointers, with generated retain/release calls to C storage. MLIR cannot reconstruct their owners from those calls. | Preserved working implementation. Requires source storage representations and compiler ABI migration; adding the buffer pass alone does not fix this boundary. |
| Source record/collection cleanup | Initialization checks and cleanup selection were embedded in semantic lowering. | Both now query ownership `places.sev`; semantic analysis supplies construction/write facts and emits hooks. Arbitrary partially initialized owning collections and full CFG initialization facts remain unfinished. |
| Native runtime C, tensor-JIT launchers and compiler tooling | Foreign ABI/resource adapters and legacy runtime implementations remain. | Preserve implementations; do not describe their presence as a C source backend or their opaque lifetimes as MLIR-verified. |

No native allocation counter or registry lookup may authorize a source transfer.
Do not remove a compatibility implementation until its source/MLIR replacement
passes alias, initialization, destruction, error-path, and sanitizer validation.

## Validation at this checkpoint

- Rust bootstrap builds the source compiler through MLIR, including a release
  profile build. A CLI regression verifies that a scalar program cannot silently
  fall back to C when the MLIR tool fails.
- 23 method regressions and 5 conversion regressions pass with optimization and
  AddressSanitizer. MIR/lowering/MLIR library checks and 26 runtime tests pass.
- The shared MLIR pipeline test checks physical allocation/release balance across
  aliases, returned buffers, branches and loops with sanitizer instrumentation.
- Source tests verify MLIR allocation/copy operations, independent storage,
  preserved raw-pointer behavior, and source-located rejection of view releases.
- The tensor registry integration currently stops before ownership lowering:
  structured `StorageView(FromElements)` generation is unsupported. It is not a
  passing end-to-end ownership test.

The source compiler's unoptimized development build exceeded the test harness's
60-second compilation budget. Source checks use the release build with an
explicit 180-second budget; this does not establish a 60-second cold-build target.

## Cold-compilation follow-up

The installed compiler rebuilt by `sev update` used the package's development
profile. A fresh hello-world compilation spent 229 seconds in semantic analysis;
a debugger sample found declaration retention during prelude generic checking.
Concurrent invocations waited on the active compilation's cache lock. This was
active CPU work, not evidence of a deadlock left by an interrupted test.

Update now selects the release profile and requires its cold smoke test to finish
within 90 seconds, restoring the previous compiler and stopping the smoke-test
process group on timeout. Source semantic lookup indexes retain names and operator
symbols once, then copy only matching declarations through normal storage. They
extend when declarations are registered and remain separate for isolated analyses.
The measured release candidate took 50 seconds for a cold hello-world run and
passed all three getting-started test files. Cold compilation remains expensive;
these changes do not complete prelude artifact reuse or the ownership migration.
Cache misses now print the source being compiled to stderr.
