# Compiler resource guards

Compiler commands need budgets **before compilation starts**. A timeout inside
the generated test executable cannot protect the compiler that builds it.

The source package testing framework now applies `[test].memory-max` and
`[test].timeout-seconds` to compilation units and test executables. The shared
hosted helper is `library/system/process/src/resources.sev`. Native pipeline steps
have their own `memory_bytes` and `timeout` budgets, including rebuilding the
compiler. Configuration for the example pipeline lives in
`sev_compiler/package.toml` under `[package.metadata.pipeline]`. Its complete example-suite test has a
separate 1,800-second outer ceiling because it runs many individually guarded
steps (30 seconds per example command, 180 seconds for the compiler rebuild).

The Python acceptance framework (`MigrationCase` and `bootstrap_mlir.run`),
the example auditor, and `tests/sev_compiler/run.sh` also guard their compiler
commands. Defaults are one job, 3,000,000,000 bytes and 60 seconds; the example
auditor defaults to 30 seconds. The shell runner and auditor refuse concurrent
budgets totaling more than 6 GB. For a compiler rebuild, use a **serial** 6 GB,
180-second budget. Environment variables can lower acceptance-runner budgets:

```sh
SEVERIAN_TEST_MEMORY_BYTES=3000000000 SEVERIAN_TEST_TIMEOUT_SECONDS=30 \
  python3 tests/sev_compiler/explicit_ownership.py ExplicitOwnership -v

python3 tests/sev_compiler/examples.py docs/examples/06-ownership \
  --timings --jobs 1 --timeout 30 --memory-bytes 3000000000

python3 tests/sev_compiler/test_resource_guard.py -v

python3 tests/sev_compiler/resource_guard.py --timeout 180 \
  --memory-bytes 6000000000 --report /tmp/compiler-build.json -- \
  sev_rust build sev_compiler --bin sev_compiler --profile release \
  -o sev_compiler/package.pkg/host/dev/bin/sev_compiler
```

`prlimit` installs a hard, inherited address-space limit and disables core
dumps. Native pipeline commands use GNU `timeout` with a two-second kill grace.
The Python watchdog kills the process session on timeout, cancellation, or an
aggregate RSS breach, including test children that create their own process
groups. It cleans up remaining session children after successful commands too.
An additional native timeout remains active if the Python supervisor exits.
Never increase parallelism and per-command budgets independently.

These are Linux hosted guards, not a container memory reservation: an address
space limit applies to each process. The Python watchdog additionally samples
session RSS every 50 ms; shared pages can be counted more than once and brief
peaks between samples can be missed. Commands that deliberately create a new
session require a cgroup for full descendant accounting. Keep headroom for the
OS and other applications. A command failing under an address-space cap is a
failure; it is not automatically labeled an observed RSS-limit breach.

Native CPU time and maximum RSS come from `/usr/bin/time`, separately from
compiler stderr. Maximum RSS is the maximum for a command and its waited
children, not their sum. Reports also include sampled session RSS, the actual
budgets, compiler stage timings, exit codes, and raw resource files. Measurements
unavailable after a forced kill remain unavailable. Failed compilations and
absent tests never count as passing performance checks.

`sev_compiler/tests/resources/src/limits.sev` exercises the native testing helper and
pipeline against success, failure, timeout, and allocation exhaustion. The
Python guard tests also verify descendant cleanup and resource measurements.
`sev_compiler/tests/resources/src/compilation.sev` compiles the ownership builder using
the native source compiler under a 15-second, 2 GB budget. Override
`SEVERIAN_SOURCE_COMPILER` to check a candidate compiler before installing it.

Run the native guard suite through its package so imports resolve:

```sh
python3 tests/sev_compiler/resource_guard.py --timeout 180 \
  --memory-bytes 6000000000 --report /tmp/native-guard-tests.json -- \
  sev_rust test sev_compiler/tests/resources
```

The Rust seed builds this native test harness; its compilation-budget test
executes the source compiler binary. Standalone `sev test FILE.sev` commands
should also use the outer wrapper. Package settings apply to `sev test PACKAGE`,
including its dependency compilation units.
