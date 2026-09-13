# Package quality pipeline

`sev build`, `sev check`, `sev run`, and `sev test` validate package policy before compiling or reusing an artifact. Manifests are JSON5 in `package.json`; generated locks and reports are JSON. `sev options` prints the complete option catalog. `sev new` and `sev init` include documented example behavior and executable tests.

## Test quality and coverage

`sev test` enables coverage by default. `sev test --coverage` explicitly enables it for the invocation. The default minimum is 91 percent for line, branch, function, and condition coverage. Each decision and boolean expression records its actual true/false outcome. Empty test collections and tests that execute no instrumented production region fail.

Test execution also fails on unused symbols (L0004), unreachable code (L0005), ineffective tests (L0009), and constant or unfinished callables (L0010). A test must assert observable production behavior. Calling a constant-returning helper does not qualify. Test-only references cannot make binary production callables reachable. Exported library APIs remain legitimate production entry points.

```json5
{
  test: {
    coverage: true,
    'coverage-line-min': 91,
    'coverage-branch-min': 91,
    'coverage-function-min': 91,
    'coverage-condition-min': 91,
    'coverage-exclude': ['tests/', 'test/', 'generated/'],
    'allocations-max': 10000,
    'memory-max': '4GB',
    'timeout-seconds': 60,
    leaks: 'deny', // allow | warn | deny
  },
}
```

Tests and the executable `main` harness are excluded from production coverage. Thresholds apply to the current package. Dependency source hits may establish that a test executed production behavior. Precompiled foreign code has no Severian source regions.

Every invocation writes a separate `package.pkg/debug/coverage/run-*/` directory containing `coverage-map.json`, `coverage.hits`, runtime records, and `summary.json`. Compiler-cache and realization inventories retain and verify the coverage map alongside its executable. Missing or damaged metadata cannot silently satisfy policy. Release builds do not add test coverage probes.

Allocation and retained-byte measurements cover the `core.memory` native boundary, including buffers released by MLIR ownership lowering. Foreign allocators that bypass this boundary are not measured. `leaks: 'deny'` fails when a test retains allocations made through that boundary; long-lived caches may need an explicit package policy.

## Documentation

L0011 checks public callable purpose, parameter names and descriptions, Returns, Errors, and Complexity. L0012 checks public class purpose, Responsibilities, and Invariants. Both default to warnings and accept the normal severity overrides and suppression directives.

Bracketed types add checks against the current signature:

```sev
# Doubles a supplied integer.
# Parameters:
# - value [int]: Number to double.
# Returns [int]:
# - Twice the supplied number.
# Errors [None]:
# - None.
# Complexity:
# - Runtime: O(1). Space: O(1).
def doubled(value: int) -> int:
    return value * 2
```

Renaming a parameter invalidates its old documentation. Explicit documented parameter and return types must agree with the signature; named error types must occur in the result union. Unannotated prose remains readable during migration. Documentation cannot prove a prose claim or an asymptotic complexity bound.

`sev check PACKAGE --emit docs -o api.md` emits API signatures and documentation. `--emit editor` emits the resolved symbol snapshot consumed by tooling.

## Editor and debugger

The VS Code extension uses compiler-resolved declaration identities for definitions, references, implementations, rename, and call/type hierarchies. Hover shows signatures and documentation; inlay hints show variable types. The extension also supplies semantic highlighting, deprecated declaration highlighting (`@deprecated` in documentation), inline lint diagnostics, unused-code dimming, and coverage gutters.

Snapshots apply to saved text. Editing invalidates them rather than presenting locations from an older document. Unicode scalar compiler spans are converted to the editor's UTF-16 positions. Coverage gutters select the latest completed report per package so historical hits cannot conceal a new regression.

`sev debug PACKAGE` builds the development profile and launches GDB. VS Code uses GDB's native DAP interpreter for breakpoints, stepping, variables, watches, and stack frames. Configure `severian.debugger` when GDB is not on PATH.

Native debug metadata currently describes scalar arguments and locals. A companion `<name>__ownership` variable exposes compiler ownership state: 0 owned, 1 shared borrow, 2 view, 3 mirror, 4 moved, 5 exclusive borrow. `view()` currently lowers to a shared borrow. The source compiler still rejects mirror operations requiring copy-on-write support; the debugger does not add that language feature. Aggregate and collection pretty printers are not provided.

## History, duplicates, and caching

L0013 indexes normalized callable bodies to identify duplicate implementations. Parameter names are normalized; bodies smaller than twelve classified tokens are ignored. L0014 reports files changed more often than `lint.churn-commits` within the explicit `quality.revision-window`. The revision window defaults to zero, disabling Git history inspection. These heuristic rules are hints and do not rewrite code.

Quality results are cached by source content, configuration, compiler identity, and the selected Git revision. Unchanged packages reuse the checked diagnostics. Changed packages currently rerun the package analysis; this is not a per-function incremental semantic database. `package.pkg/cache/quality/analysis.json` reports cache reuse and lint wall time.

## Regression commands

```sh
sev_rust build sev_compiler --bin sev_compiler
sev_rust test test/validation/packages/quality
python3 library/package/tests/quality.py -v
node --test editors/vscode/tests/*.test.js
cargo test -p severian-driver config::tests --lib
```

The CLI suite compiles real programs, compares golden diagnostics and coverage counts, mutates an implementation to require an assertion failure, verifies source navigation, exercises native allocation tracking, and runs GDB against a compiled program. Its debugger checks require local ptrace access. No test is skipped when a required tool or execution capability is missing.

Validation completed with 18 policy regressions through the Rust seed, the 16-test CLI suite plus the added ownership-transition regression, editor tests, and three Rust configuration tests. The existing ignored Rust mirror test remains unchanged. After the final debug-storage change, both native debugger regressions and the coverage pass/fail regression were rerun successfully.

Executing the policy test package itself through the source compiler currently stops at E000124 package-import errors (dependency identity or the lexer's relative import of universal character encoding). The Rust-seed policy suite and source-compiler CLI fixtures are executable and passing; full self-hosting of that policy package is not claimed here.
