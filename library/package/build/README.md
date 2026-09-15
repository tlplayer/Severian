# package.build

Compiler pipeline coordination and shared build/document contracts. This package
depends on `package.cache` for verified storage and `package.profile` for command
measurements. It has no dependency on compiler AST, semantic, MIR, or MLIR types.

`CompilerPipeline(root, enabled=true)` passes `CompilerStepResult` artifact
references between `compiler_step` calls. Each step key includes its explicit
version, tool identity, input digest, options, declared extra inputs, working
directory, and relevant toolchain environment. Successful outputs become cache
blobs. If changed upstream input produces the same output bytes, downstream
steps see the same digest and can reuse their results. Broken records or outputs
request execution again; failed execution cannot publish a completion record.

The source compiler uses these boundaries for ownership lowering, LLVM dialect
lowering, LLVM IR translation, native object generation, and bytecode output.
C preprocessing reruns on a compilation-unit miss to discover transitive header
changes; unchanged preprocessed bytes reuse their object. Native linking remains
conservative and reruns after a unit miss. Unchanged complete units bypass the
whole pipeline through the package compilation-unit cache.

This is not yet per-function semantic caching: a changed source input still
re-enters parsing and semantic analysis for its compilation unit. Generic
specializations are generated in their consumer unit. Debug source mappings are
part of emitted IR, so source-location changes can prevent convergence.

Generated output is owned by the invocation directory's `package.pkg`:

```text
package.pkg/
  bin/<target-name>             latest selected runnable build
  artifacts/<platform>/<profile>/<identity>/
  dependencies/<package>/<content-id>/
  build/<identity>/            unit state and pipeline evidence
  cache/compiler/              stage records and shared IR/object blobs
  debug/quality/               lint and file-contribution reports
  debug/coverage/              runtime coverage reports
  debug/profile/               frontend and native-stage measurements
```

Explicit `-o` still controls delivery. Platform/profile identities remain in
artifact storage; `bin/<name>` exposes the latest selected build. Wildcard-import
restrictions in `package.json` are deferred.
