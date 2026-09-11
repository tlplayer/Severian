# Source compiler collection and prelude validation

Validated on 2026-09-11 using the rebuilt `sev_compiler`. Rust was used to
build the source compiler; the new generic and prelude gates compile their
subjects with `sev_compiler` and execute the resulting native code.

The baseline is commit `6728cd82`, with the workspace-only native-discovery
repair applied to both builds so the comparison reaches compilation.

| Examples | Baseline (190) | Final (191) |
| --- | ---: | ---: |
| Build passed | 95 | 96 |
| Build failed | 95 | 95 |
| Run passed | 95 | 96 |
| Run blocked by build | 95 | 95 |
| Test passed | 77 | 78 |
| Test failed | 92 | 92 |
| No tests declared | 21 | 21 |

All 191 examples were audited. The new container-provider example passes
build, execution and tests. The initial sweep caught the closures example's
reserved free `view` declaration; it was renamed `borrowed_identity` and its
build, execution and tests were rerun successfully. After that correction,
all 190 existing examples retain their baseline statuses. Existing failures
remain; this is not a claim that the entire example suite passes.

Additional checks:

- Eight collection-generic test groups passed, covering named arguments,
  constructor bindings, aliases, traits, mixed layouts and union values.
- Five prelude-policy groups passed, including native overloads, reserved-name
  errors in build/check/test, package execution, exclusions and restoration
  of reservations after a cached build.
- Four adjacent package-library tests passed, including policy ownership.
- Seven provider/grammar fixture tests passed. Two stale provider fixtures
  also failed on the baseline; they were corrected to reflect the source-owned
  list alias and to update dependent code when renaming the list growth API.
- Both changed legacy generic diagnostic gates passed in build and check.
- `git diff --check` passed.

Commands include `tests/sev_compiler/collection_generics.py`,
`tests/sev_compiler/prelude.py`, `sev_rust test library/package` and
`tests/sev_compiler/examples.py`. Set `SEVERIAN_SOURCE_COMPILER` for the Python
native gates; use `--compiler` for the example runner. Provider fixture checks
were targeted unittest methods, not a claim that every legacy gate was run.

The full audit artifacts are in `/tmp/sev-collections-examples-prelude` and
the closures rerun in `/tmp/sev-collections-examples-prelude-closures`.
The combined results are `/tmp/sev-collections-work/examples-validated.json`.

Production set/map provider extraction, default container parameters and
variadic constructors remain in the implementation plan. Excluding the
compiler-handled intrinsics listed in `universal/prelude.toml` remains an
explicit unsupported diagnostic.
