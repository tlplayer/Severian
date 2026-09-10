# Source compiler profiling checkpoint

Rebuild the source compiler from the repository root:

```sh
sev_rust build sev_compiler --bin sev_compiler
python3 tests/sev_compiler/memory_ownership.py -v
python3 tests/sev_compiler/known_good_spine.py
python3 tests/sev_compiler/owned_records.py -v
```

Use one timing file per compiler invocation. Stage times are seconds measured
with the monotonic clock, and are separate from stdout and emitted IR:

```sh
SEVERIAN_TIMINGS=/tmp/basic.tsv sev build \
  docs/examples/02-functions/01-basic/01-basic-functions.sev \
  --emit mlir -o /tmp/basic.mlir
```

Stages include prelude loading, subject loading, semantic analysis, MIR, MLIR
construction and printing. Layout-query programs additionally report target
layout discovery. Agent IR emission has its own stage. Completed stages are
flushed even if a subsequent stage fails. These timings exclude native LLVM
lowering, linking, execution and process startup.

Audit examples with independent build, execution and test results:

```sh
python3 tests/sev_compiler/examples.py \
  docs/examples/01-types docs/examples/02-functions/01-basic \
  docs/examples/03-testing docs/examples/06-ownership \
  --timings --jobs 1 --timeout 45
```

The report includes total command times, compiler stage times, diagnostics and
paths to raw logs. Failed and absent tests are not counted as passes. Use one
job for performance comparisons; multiple jobs are useful for correctness
audits but introduce contention. Do not compare `--emit agent-ir` times against
native/MLIR emission: agent IR intentionally retains the complete graph.

## Measured improvement

On this checkout's host, five serial compilations of the basic-functions
example had median wall times of **2.464 s before** and **1.084 s after** the
change (2.27× faster). Both compiler binaries read the same current source
providers. The emitted MLIR shrank from 15,595 to 247 lines. Median peak RSS
fell from 1,410,328 KiB to 473,968 KiB, a 66% reduction. These measurements
include retaining all native ABI declarations for conflict checking.

The compiler now lowers only reachable prelude function bodies. C ABI
declarations remain present so conflicting signatures are still diagnosed.
Every application
declaration remains a root, preserving library exports and diagnostics in
unused application functions. Required calls are collected from lowered CFGs,
including calls introduced by inlining and source control grammars. Recursive
and forward calls are processed once. Semantic analysis still checks the full
prelude; that is now the largest measured stage in this small example.

Local measurement artifacts, commands, binary hashes and individual samples
are in `sev_compiler/package.pkg/profiles/reachability-final/results.json`.

## Validation

The rebuilt compiler passed 126 acceptance checks and all 80 targeted tests
across memory ownership, the known-good spine, owned records, source contracts,
enums, list growth, aggregate buffers, type semantics and declaration grammar.
Memory checks include AddressSanitizer and balanced allocation accounting.

The full 111-file audit reports 62 successful native builds/runs, 49 build
failures, 46 passing test runs, 55 failing test runs and 10 files without tests.
The layout and resource-lifetime examples in `01-types/04-memory` pass; the raw
allocation example remains a failure. `06-ownership/05-drop.sev` also passes.
The report and per-command diagnostics are in
`sev_compiler/package.pkg/examples/memory-profile-current/REPORT.md`.
Acceptance and regression logs are copied into the profiling artifact directory.

## Remaining scope

This checkpoint does not complete the example inventory. Raw `allocate`/`free`
and `unsafe` scopes, explicit borrow/move/clone semantics, general resource
transfers, resource cleanup in loops/errors, and several advanced function,
generic and test modes still need implementation. Supported local resource
cleanup is checked separately from existing SSA buffer deallocation.

The full example report is generated under
`sev_compiler/package.pkg/examples/`; use its individual diagnostics to select
the next implementation work. Never replace unsupported examples with empty
programs or treat unsupported tests as successful.
