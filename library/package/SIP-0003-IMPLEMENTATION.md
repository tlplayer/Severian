# SIP-0003 build integration

Native library retention now connects compiler-produced objects and archives to
`package.interface` and `package.metadata`. Completed realizations use:

```
<package>/package.pkg/
  artifacts/<target>/<build-id>/object/*.o
  artifacts/<target>/<build-id>/archive/*.a
  artifacts/<target>/<build-id>/dynamic/*.so
  artifacts/<target>/<build-id>/ir/
  package.pkgi/severian/<build-id>/lib.sevi
  metadata/realizations/<build-id>.json
  metadata/{symbols,types,layouts,dependencies}/<build-id>.json
  build/<build-id>/
  bin/<target>/<build-id>/<name>
  debug/quality/lint/
  source/
```

`lib.sevi` binds exports to typed realization metadata and checksummed native
artifacts. The compiler consumes its declaration contracts. Metadata sections
are derived views of that realization. Scalar exports have native symbol and
callable-type records; layouts are not invented for unsupported aggregate or
generic contracts. Already materialized dependencies retain exact build and
interface identities. The unit identity covers source-only dependencies too.

`metadata/build-records/` retains the legacy toolchain/cache inventory while the
canonical typed records occupy `metadata/realizations/`. The binary interface
sidecar remains for restoring older compilation outputs. `bin/<name>` remains
a convenience copy; it is not the versioned artifact identity.

Quality analysis belongs to `package.diagnostic.lint`, with compatibility
entry points in the package coordinator. Independent formatter and coverage
extraction, aggregate/generic native ABI exports, and rebuilding the installed
compiler remain migration work.

A native geometry realization was built from compiler-emitted MLIR using an
explicit native export step, then MLIR/LLVM/Clang/archive/shared-link stages.
Its build provenance records that manual backend path. It is evidence of real
native payloads, not a claim that the installed `sev build` already executes the
new retention code. The installed compiler currently fails on source-language
compatibility and its deleted quality-runtime input.
