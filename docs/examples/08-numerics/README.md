# Numerics examples

From the repository root, build and run a program with the source compiler:

```sh
bin/sev_compiler build docs/examples/08-numerics/01-numeric-types.sev -o /tmp/numeric-types
/tmp/numeric-types
bin/sev_compiler test docs/examples/08-numerics/01-numeric-types.sev
```

Most files express their behavior in `test` blocks, so building their program
entry alone does not validate their numerical operations. Audit every file's
native build, program execution, and tests with:

```sh
python3 tests/sev_compiler/numerics_examples.py
```

The audit uses the already-built `sev_compiler` exclusively. It continues after
failures, returns nonzero if any example fails, and keeps its JSON report,
stdout/stderr logs, MLIR, LLVM IR, and executables in
`sev_compiler/package.pkg/numerics/run-*`. The GPU profiling test is explicitly
skipped by its source declaration unless that declaration is changed for a
configured backend; a successful host fallback does not demonstrate GPU use.

Focused compiler regressions run with:

```sh
python3 tests/sev_compiler/numerics.py -v
```

The scalar examples cover `f32` and `f64`, explicit conversions, mixed numeric
promotion, and hosted math. Tensor and placement examples also require shape
packs, dimension constraints, tensor operation lowering, and the execution
boundaries described in `15-tensor-exhaustive.sev`. These remain failing work
in the active source-compiler pipeline; the audit does not exclude them or
substitute results from the Rust seed.
