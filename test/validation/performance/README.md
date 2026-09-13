# Compiler performance regressions

Run from the checkout, with the native compiler installed:

```sh
bash test/validation/performance/bootstrap_freshness.sh
bash test/validation/performance/native.sh
bash test/validation/packages/native_artifacts.sh
bash test/validation/performance/libraries.sh
```

The shell tests invoke concrete compiler executables. `SEVERIAN_SOURCE_COMPILER`
selects a candidate native binary; `SEVERIAN_BOOTSTRAP` selects the Rust bootstrap.
Generated test projects stay in temporary directories. Measurements and logs go
to `package.pkg/debug/profile/run-<id>/`; a successful run atomically updates
`package.pkg/debug/profile/baseline.json` with cold and three warm build times.
Debug output is local and is excluded from published packages.
Default gates: semantic analysis under 12 seconds, warm native builds under
2 seconds, warm bootstrap builds under 1 second. Slower CI machines may explicitly
set `SEVERIAN_SEMANTIC_SECONDS`, `SEVERIAN_WARM_BUILD_SECONDS`, or
`SEVERIAN_BOOTSTRAP_WARM_SECONDS`; do not raise limits to conceal regressions.

`test with profile ... with { defer time < 1s, ... }` checks resource predicates
at completion of each native test case, including cases that return early.
`library/testing/src/profile.sev` supplies lifecycle adapters; reusable
`Measurement`, `measure`, `time`, `allocations`, and `memory` operations live in
`library/system/profiling/src/measurement.sev`. Measurements use the statistics
API in `library/core/storage/src/statistics.sev`, not private memory bindings.
`time` is monotonic elapsed time. `allocations` counts successful source buffer
and reference-counted storage allocations in the executing thread. `memory` is
cumulative buffer allocation bytes plus storage payload bytes, including those
later freed. It excludes ownership registry overhead, foreign libraries' private
allocations, and other threads; it is not RSS. The providers use disjoint
allocation paths, and the C tests check their combined totals and that retaining
or releasing an owner does not count as another allocation.
The negative-budget test proves resource predicates are enforced.

The file fixture calls `library/system/file/src/text.sev`: `temporary`, `write`,
`read_checked`, and `remove`. `read_checked` returns `ReadResult(contents, error)`;
zero error distinguishes successful empty reads from failures. The library owns
NUL marshalling, pointer conversion through core.memory, and release of successful
provider reads. The fixture has no C or MLIR declarations. The older typed File
dispatch API remains in `src/lib.sev`; this hosted source-string module does not
change that API's error-union contract or the bootstrap's string representation.

Set `SEVERIAN_PROFILE_ACTIVE=1` to print frontend, MIR, and backend wall times.
Bootstrap MIR passes and external backend tools report their own durations.
Parent stages include child stages: do not add both when calculating totals.
Bootstrap freshness records and smoke verification receipts use JSON. An
unchanged update validates content hashes before skipping compilation and reuses
a successful smoke receipt only for the same compiler and build input inventory.

Additional profile cases live alongside the lexer, string, file, and MLIR source
tests. The full string suite and the documented profile example pass. Broader
source suites currently have separate frontend blockers: native compilation of
the file package rejects its existing `@file` trait syntax, and native MLIR tests
reject owned buffer elements. The bootstrap MLIR suite instead stops at the
existing unchecked optional `emitted.location.line` access. Those suites are not
claimed as passing gates; use the runnable baselines above for measured results.

The `libraries.sh` runner checks the C file provider in
`library/system/file/tests/native_io.c`, ownership accounting in
`library/core/storage/tests/statistics.c`, and the JSON provider in
`library/data/json/tests/native_json.c`. The former ad hoc `native_io.c` under
this directory has been removed after splitting these checks by their owners.

Native text tests exercise `core.text` formatting, escaping, object conversion,
and captured print output. The public `.txt`/`.sev`/`.json` `file.read` dispatch
and Data indexing tests run through the explicitly named bootstrap executable;
they are not presented as passing native-frontend tests. The native frontend's
`@file` parsing limitation remains a separate unresolved issue.

JSON's retained-storage check covers all decoded strings and rows, including
failure cleanup. Its old implementation leaked rows and used untracked malloc
strings; its old allocation count therefore understates the work it did and
should not be directly compared with the new owned allocation count.
