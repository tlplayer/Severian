# package.diagnostic

Package-owned quality policy, coverage aggregation,
and build-input reporting. Compiler instrumentation supplies `CoverageRegion`
and `CoverageTest` records through `coverage_map`; this library validates runtime
records and applies package thresholds. It never lowers or rewrites compiler IR.

[package.diagnostic.lint](lint/README.md) owns lint rules, source metrics,
suppression, reporting and enforcement. The parent keeps compatibility exports.

Lint reports belong in `package.pkg/debug/quality/lint`; file contribution
reports belong in `package.pkg/debug/quality`.
Coverage results belong in `package.pkg/debug/coverage`. Timing and resource
measurements are owned by the sibling `package.profile` library.

`contributions(root, consumed, testing=false)` compares owned `.sev` files with
the compiler's recorded inputs. `report_contributions` emits `build-inputs.json`
and `B0001` informational diagnostics for unconsumed files. Reports describe the
selected targets. Test-only sources and alternate targets may still be needed;
unused-file reports are review evidence and never delete source files.

The compiler adapter supplies source mappings and instrumentation. Package
configuration, exclusions, lint rules, aggregation, and report formatting stay
here so tooling can consume the same policy without importing an IR backend.

## Import formatting

Linting formats selective imports by default before resolution and cache-key
calculation. Set `lint.format = false` to opt out. Comma-separated imports fit
on one line; longer lists use explicit `\` continuations within 80 columns.
Brace groups stay vertical, and aliases are preserved:

```sev
from build_flow import
{
    CompilerPipeline,
    CompilerStepResult,
    compiler_step,
}
```

Imports containing internal comments retain their original layout so comments
are not moved or discarded. Literal text is never formatted as code.
