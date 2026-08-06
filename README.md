# Severian

Severian is a compiled systems language with Python-like syntax, Rust-like safety,
and MLIR as the compiler backbone.

The repository is being built piece by piece around a checked, native CPU core:

- `compiler/ast`: source-level syntax tree nodes.
- `library`: official Severian packages, manifests, documentation, and
  language-native tests.
- `library/platform`: typed declarations for native services used beneath public packages.
- `docs/language`: living language notes.
- `docs/examples`: example `.sev` programs that should become compiler fixtures.
- `docs/examples/14-packages`: Cargo-like package and workspace examples.

## Design Center

Severian keeps simple code readable while giving the compiler enough structure to
infer ownership, verify memory safety, and lower predictable programs into MLIR.

The intended feel is:

- Python readability through indentation, concise declarations, and expression
  oriented code.
- Rust safety through ownership inference, explicit escape hatches, and
  recoverable errors as values.
- Go practicality through direct concurrency primitives and simple tooling.
- Cargo-style official packaging through `sev`, with one standard manifest,
  build, test, doc, and publish workflow.

## Example

```sev
def add(a: int, b: int) -> int:
    return a + b

print(add(1, 2))
```

## Native Compiler Baseline

The compiler's mandatory acceptance suite currently takes every valid source
under `docs/examples` through parsing, semantic analysis, ownership checking,
MLIR, native linking, execution, and exact stdout verification. Expected-invalid
programs must match diagnostic fixtures. The generated
[`docs/NATIVE_STATUS.md`](docs/NATIVE_STATUS.md) is the single inventory of that
evidence; specialized examples are a host-native baseline, not a claim that
their external service, accelerator, or freestanding target is production-ready.

```sh
cargo run -p severian-driver --bin sev -- check docs/examples/00-getting-started/01-hello.sev
cargo run -p severian-driver --bin sev -- compile docs/examples/00-getting-started/01-hello.sev
cargo run -p severian-driver --bin sev -- run docs/examples/00-getting-started/01-hello.sev
```

Once `sev` is installed, a source path by itself provides the Python-like
development loop while still executing compiled native code:

```sh
sev program.sev
```

`sev` compiles to a temporary executable, runs it with inherited standard
input/output, propagates a failing exit status, and removes the temporary file.
If a source has tests but no `main`, the generated native entry point executes
those tests and prints the real pass count.

`sev build` reads Cargo-compatible `[package]`, `[[bin]]`, `[dependencies]`, and
`[workspace] members` fields from `Severian.toml`. Package and workspace binaries
are emitted under `target/debug`. Path libraries are checked in dependency order
and emitted as `target/debug/deps/lib<package>.sevi`; consumers then compile from
those artifacts. Library-local tests are not linked into downstream application
test binaries. `sev build source.sev` uses the source stem as the binary name.

The CLI is also a conventional Cargo binary crate:

```sh
cargo install --path compiler/driver
sev doctor
sev --help
```

Internal compiler dependencies carry both local `path` entries and registry
versions, so the compiler crates can be published in dependency order and the
final `severian-driver` package can provide the `sev` executable through a Cargo
registry.

`compile` verifies the emitted MLIR, translates its LLVM dialect to LLVM IR, and
links a native executable named `a.out` by default. Use `-o executable` to choose
another path. `emit-mlir` prints the intermediate MLIR for inspection, while
`run` executes the validated HIR for a fast development loop.

## Example Fixtures

Every source snippet in the language docs should have a matching file under
`docs/examples`. Once the parser and driver exist, those files should be compiled
as part of the test suite.

Run the ordered native acceptance harness with:

```sh
tools/check_docs_examples.sh
```

Regenerate or CI-check the definitive per-group status table with:

```sh
python3 tools/example_status.py --write
python3 tools/example_status.py --check
```

The harness inventories every `.sev` file under `docs/examples`; there is no
curated native subset. Every valid file must provide `main()` or attached native
tests, reach valid MLIR, link, execute, and match adjacent output fixtures.
Missing executable coverage, missing output fixtures, compiler failures,
timeouts, stderr, and output mismatches all fail mandatory acceptance. Every binary
that successfully links is retained under the matching `bin/examples` path even
when its runtime or output check fails. Use `--frontend-only` for diagnostic
front-end work; it does not establish example completion.

## Example benchmarks

Root-level [`bench/`](bench/) contains equivalent Severian, Rust, and Python
programs for every example in directories `00` through `07`. The benchmark
validates exact stdout before measuring compilation and fresh-process execution:

```sh
python3 bench/run.py
```

Run `tests/check_bench_examples.sh` for the one-sample correctness gate. The
gate is exhaustive and remains red when any inventoried example is not a valid
native executable.

## Official library

The official library uses flat imports such as `import network` and
`from math import jacobian`; a full import exposes the package's available names. Its package
catalog and compiler/library/runtime
ownership boundary are documented in `library/README.md` and
`library/CATALOG.md`.

Run every library package that currently has an implementation with:

```sh
tools/check_library.sh
```

Author-facing model code uses the framework-neutral `model` namespace:

```sev
import model
from model import neuralnet as nn
```

The native transformer and OCI deployment example is under
`docs/examples/28-transformer-container`. Its host-versus-container benchmark
is run with `python3 bench/transformer-container/run.py`.

The operating-system example under `docs/examples/29-operating-system` is a
hosted kernel laboratory that exercises memory ownership, mappings, process
capabilities, a VFS, syscalls, interrupts, scheduling, and concurrent workers.
Its documentation separately identifies the compiler and runtime work required
for a genuinely freestanding boot target.
