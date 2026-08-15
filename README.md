# Severian

Severian is a compiled systems language with Python-like syntax, Rust-like safety,
and MLIR as the compiler backbone.

The repository is being built piece by piece around a checked, native CPU core:

- `compiler/ast`: source-level syntax tree nodes.
- `library`: official Severian packages, manifests, documentation, and
  language-native tests.
- `library/ffi`: stable types used by package-owned foreign interfaces.
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

`sev build` reads Cargo-compatible `[package]`, every `[[bin]]`, `[lib]`,
`[dependencies]`, and nested `[workspace] members` fields from `package.toml`.
`sev init` and `sev new` generate a self-documenting manifest containing every
supported project/build control. Fresh projects explicitly use permissive
coverage, memory, architecture, and type-safety values so teams can tighten the
ratchets as the codebase matures. `[build].diagnostics = "user"` is the clean
source-oriented default; set it to `"internal"` (or pass
`--diagnostics=internal`) when investigating compiler/backend internals. This
mode only adds implementation metadata: source errors, runtime details, and
automatic native crash stacks are complete on the first normal invocation.
Package and workspace artifacts are emitted under `target/debug`. Path libraries are checked in dependency order
and emitted as `target/debug/deps/lib<package>.sevi`; consumers then compile from
those artifacts. Library-local tests are not linked into downstream application
test binaries. `sev build source.sev` uses the source stem as the binary name.
Before emitting artifacts, every build runs the complete manifest policy in
order: compile, architecture dependency/layer rules and file budgets, unit tests, profile tests, coverage,
memory/leak checks, and integration tests. Those gates cannot be removed with a
CLI skip flag or omitted from a custom pipeline. Standard and path dependencies
are not charged to the consuming package's coverage percentage. See
[`docs/BUILD_POLICY.md`](docs/BUILD_POLICY.md) for the manifest schema.

The CLI is also a conventional Cargo binary crate:

```sh
cargo install --path compiler/driver
sev doctor
sev --help
```

### Step 3: standard CLI and editor surface

The standard single-file commands are:

```sh
sev run main.sev
sev build
sev test
sev test --profile
sev test --profile --memory
sev test --profile --memory --leaks
sev architecture
sev architecture --graph
sev debug main.sev
sev clean
```

Inside a project, the manifest supplies the target:

```sh
sev run
sev build
sev test
sev test --profile
sev test --profile --memory
sev test --profile --memory --leaks
sev debug
```

`run` builds and executes, `build` reports the project's compiler diagnostics,
and `test` runs all native tests and their contracts. `test --profile` selects
only profile tests and enforces their time, memory, allocation, and runtime
contracts. Every profile test prints elapsed nanoseconds, cumulative bytes
allocated, and allocation count. Add `--memory` to run those tests under native
memory and undefined-behavior sanitizers; add `--leaks` for opt-in leak
detection. `debug` creates an unoptimized native build with debug symbols and
launches LLDB or GDB; `SEVERIAN_DEBUGGER` selects another debugger. `clean`
removes only the resolved project's generated `target` directory.

The VS Code Run, Build, Test, Profile, and Debug actions invoke those same CLI
commands. The extension does not implement a separate editor-only execution or
profiling path.

### Accelerator kernel backends

`sev kernel inspect` explains whether a tensor kernel uses the portable
StableHLO/XLA path or a specialized Triton GPU path. `sev kernel emit` writes
the selected standalone artifact; it does not generate benchmark adapters or
encode a harness protocol. See [docs/KERNEL_BACKENDS.md](docs/KERNEL_BACKENDS.md).

`sev build` collects independent diagnostics across package sources before it
emits artifacts. Use `--max-errors N` to bound the batch and
`--message-format json` for editor, CI, or automated-repair integrations. A
direct function-only `.sev` file compiles as a linkable module rather than
requiring an artificial `main` function.

Compiler IR invariants are checked after resolved HIR, package linking, every
HIR transformation, and MIR construction. `sev build --verify-each` prints the
verified boundaries for compiler-development and pass-bisection logs; verifier
failures name the transformation that first produced invalid IR.
The staged refactor and its strong-ID migration order are documented in
[compiler architecture](docs/COMPILER_ARCHITECTURE.md).

### Naming lint

`sev lint [path]` enforces naming by semantic role: variables, functions,
modules, packages, and decorators use `snake_case`; types use `PascalCase`;
and constants use `SCREAMING_SNAKE_CASE`. `sev lint --fix [path]` applies only
collision-free file-wide renames and direct compatibility spelling fixes.
External member names are diagnosed but are not automatically rewritten.

Prefer one clear word where it is unambiguous; otherwise use `snake_case`.
CamelCase APIs remain callable only for compatibility and are linted in
Severian source. Dynamic field selection uses `object.get(field)` and
`object.set(field, value)`; fixed fields continue to use `object.field`.

Ordinary words stay complete (`system`, `implement`). A small explicit registry
preserves established technical and scientific spellings such as `XLA`,
`StableHLO`, `ReLU`, and `GELU`. See [docs/NAMING.md](docs/NAMING.md).

### Test assurance

Severian can measure whether native tests reach production source, whether those
tests detect small semantic changes, and whether the exercised native runtime
triggers compiler sanitizers:

```sh
sev coverage path
sev test path --mutate
sev test path --mutate --limit 20
sev test path --memory
sev test path --profile --memory
sev test path --profile --memory --leaks
sev memory path
sev memory path --sanitizer thread
sev memory path --leaks
```

`sev coverage` reports line, statement-region, executable-branch, and function
coverage for every valid `.sev` source discovered below `path`. It executes
ordinary native tests, excludes integration-test bodies from the numerator, and
leaves main-only programs visibly uncovered. Machine-readable
`coverage-report.json`, `coverage-map.json`, and consolidated `coverage.hits`
files are written under `target/coverage`. A directory run keeps collecting
after a broken target and lists every source it could not compile or execute.

Mutation testing changes typed HIR after semantic and ownership checking, then
rebuilds and runs the original tests. Its initial deterministic operators cover
arithmetic, comparison, Boolean, and logical changes. Compile-invalid mutants
are reported separately rather than counted as killed; surviving mutants still
need human review because some changes are semantically equivalent.

`sev test --memory` and `sev memory` default to AddressSanitizer plus
UndefinedBehaviorSanitizer. Combining `--profile --memory` focuses the run on
profile tests while preserving their speed and allocation report. This makes a
single development run useful for performance regressions, invalid memory
access, undefined behavior, and (with `--leaks`) leaked allocations.
ThreadSanitizer and MemorySanitizer run alone because their runtimes are not
compatible with the default pair. Leak checking is opt-in because the current
native value runtime intentionally retains process-lifetime allocations and
some restricted environments cannot run LeakSanitizer.

The guarantees and limits of these checks are documented in
[the Severian memory model](docs/MEMORY_MODEL.md).

Internal compiler dependencies carry both local `path` entries and registry
versions, so the compiler crates can be published in dependency order and the
final `severian-driver` package can provide the `sev` executable through a Cargo
registry.

`compile` verifies the emitted MLIR, translates its LLVM dialect to LLVM IR, and
links a native executable named `a.out` by default. Use `-o executable` to choose
another path. `run` always builds and executes that native artifact; it never
executes HIR directly.

Every active compiler representation can be inspected through the same command
surface:

```sh
sev --emit hir source.sev
sev --emit mir source.sev
sev --emit mlir source.sev
sev --emit llvm source.sev
sev --emit asm source.sev
sev --emit stablehlo xla_tensor_source.sev
```

StableHLO is emitted only for tensor functions supported by the XLA path.

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
`from math import jacobian`; nested shipped packages use names such as
`import model.speech`. Official packages never appear in an application's
`[dependencies]` table. A full import exposes the package's available names. Its package
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

The operating-system work under `docs/lab/operating_system` is an experimental
kernel laboratory for memory ownership, mappings, process capabilities, a VFS,
syscalls, interrupts, scheduling, and concurrent workers. Proven components can
be deliberately ported into production packages; incomplete experiments remain
isolated in the lab.

# Language Fundamentals

Severian's surface syntax is intentionally familiar to Python programmers, but
the compiler treats the program as a statically checked systems language.

## Bindings

Local bindings are stable by default. `:=` creates a changeable local binding
that can be assigned again.

```sev
name = "Ada"
count := 0

count += 1
```

Use `:=` when the binding itself changes over time: counters, builders,
accumulators, state machines, and buffers. It does not mean dynamically typed;
types are still inferred or checked statically.

Plain `=` bindings are stable and cannot be reassigned. `:=` is the explicit
form for changeable bindings.

```sev
int MaxRetries = 3
float Pi = 3.1415926
```

Explicit types are available where they clarify public APIs or interop.
Valued declarations use one concrete type before the name.

```sev
int width = 1920
int height = 1080
```

Triple double quotes create block strings. Embedded newlines and quotes are
preserved as string data:

```sev
message = """first line
second "quoted" line
"""
```

Uninitialized fields use `name: Type`, because class schemas tend to evolve and
the name is the stable part of the declaration.

Class-like types use PascalCase, including `Result`, `Option`, `Channel`, and
`Buffer`. Ubiquitous primitives such as `int`, `float`, and `string` remain
lowercase. Parameterized types follow Python's square-bracket convention.
Parentheses are reserved for calls and runtime construction.

### Tensor types

Tensor annotations retain their scalar type and, when supplied, every ranked
dimension in HIR. Dimensions may be static integers or `dynamic`:

```sev
def encode(input: Tensor[f32, dynamic, 768]) -> Tensor[f32, dynamic, 768]:
    return input
```

The tensor type system distinguishes `f32`, `f64`, `i32`, and `i64`. Omitting
dimensions, as in `Tensor[f64]`, denotes dynamic rank. Ranked types are checked
for scalar, rank, and compatible static dimensions at call boundaries. The
current executable `tensor.ranked` constructor and linalg kernels use `f64`;
the other scalar kinds are retained through typed HIR but do not yet have their
dedicated unboxed runtime ABI and kernels.

### Naming

Severian uses snake case for values and callables, reserving PascalCase for
named types and screaming snake case for constants.

| Role | Convention | Examples |
| --- | --- | --- |
| variables, parameters, fields | `snake_case` | `token_count`, `hidden_state` |
| functions and methods | `snake_case` | `load_model`, `matrix_multiply` |
| types, classes, traits, variants | `PascalCase` | `TensorShape`, `TransformerBlock`, `HttpServer` |
| constants | `SCREAMING_SNAKE_CASE` | `MAX_THREADS`, `DEFAULT_BUFFER_SIZE` |
| packages, modules, import aliases | lowercase `snake_case` | `tensor`, `safe_tensor`, `distributed_system` |

Canonical mathematical and ML names preserve established notation when the
name itself matters: `ReLU`, `GELU`, and `LSTM` are valid named operators or
types, while ordinary callable APIs remain `relu()` and `gelu()`.

Acronyms should remain readable without becoming acronym soup. A single acronym
may fuse with its semantic word (`httpserver`), while adjacent acronym concepts
use boundaries: `http_tps_server` and `xla_gpu_client`, not `xlagpuclient`.
Standard domain names such as XLA, GPU, HTTP, MLIR, HIR, and IR are acceptable;
ordinary words stay complete: `statement`, `expression`, `platform`, and
`configuration`, not arbitrary compressed forms.

There is no mixed-case coordinate exception. Use `point.x` for a fixed field,
or `point.get(axis)` and `point.set(axis, value)` when the field name is a
variable. Compatibility camelCase remains callable but receives a lint warning.

Primitive types remain lowercase: `int`, `float`, and `string`.

The linter reports names that do not follow the convention. A variant arm may
omit its field list; the fields declared by that variant are then bound under
their declared names for the arm's scope.

## Control Flow

`while` keeps the condition next to the keyword. A scoped setup clause can follow
the condition with `with`.

```sev
while count < 3 with count := 0:
    print(count)
    count += 1
```

The `with` setup runs once before the first condition check. Names introduced by
the setup live only inside the loop condition and body.

`for` loops accept any collection-producing expression. `range` has the same
one-, two-, and three-argument forms as Python, including negative steps.
`enumerate` and `zip` produce tuple elements that can be destructured directly.

```sev
for index, value in enumerate(values):
    if value < 0:
        continue
    if index == 10:
        break

for index in range(size(values) - 1, -1, -1):
    print(values[index])
```

Conditions use `else condition:` branches and also support chained comparisons,
`in`, and `not in`. The Python-compatible `elif` and legacy `else if` spellings
remain accepted with lint warnings during migration.

## Collections And Expressive Iteration

Lists and strings support negative indexing and `[start:end:step]` slices.
Omitted bounds and negative steps follow Python's indexing rules. Native string
length, indexing, slicing, and character iteration operate on Unicode code
points consistently.

```sev
tail = values[-1]
middle = values[1:-1]
evens = values[::2]
backwards = "aé🙂"[::-1]
```

List, set, and map comprehensions accept destructuring, filters, and multiple
`for` clauses.

```sev
sums = [left + right for left, right in zip(xs, ys)]
products = [left * right for left in xs for right in ys if left != right]
remainders = {value % 3 for value in range(10)}
squares = {value: value * value for value in range(10)}
```

Expression lambdas use `|parameters| expression`. They can capture surrounding
bindings and drive collection transforms and key sorting.

```sev
offset = 3
shifted = values.map(|value| value + offset)
positive = shifted.filter(|value| value > 0)
total = positive.reduce(|sum, value| sum + value, 0)
shortestFirst = words.sorted(|word| size(word))
```

Lists also provide deque operations (`appendleft`, `popleft`) and min-heap
operations (`heapPush`, `heapPop`). Maps provide `get` and `setDefault`; sets
provide union, intersection, difference, and symmetric difference.

## Functions

Functions use Python-like `def` syntax with optional return annotations.
Parameters use `name: Type`, which keeps the parameter name fixed while richer
accepted input types grow to the right.

```sev
def add(a: int, b: int) -> int:
    return a + b

test:
    assert(add(1, 2) == 3)
```

Functions may be declared inside another function. The inner function is a
closure and can read bindings from the enclosing function.

```sev
def shifted(value: int, offset: int) -> int:
    def add_offset(current: int) -> int:
        return current + offset
    return add_offset(value)
```

Use `|` for accepted type alternatives.

```sev
def parse(value: string | int | float) -> float:
    return float(value)
```

Tests can be attached directly to functions. They compile with the function and
can call it without extra ceremony.

```sev
def x() -> int:
    return 0

test:
    assert(x() == 0)
```

A `test:` block attaches to the immediately preceding function or constructor at
the same indentation level. Inside a class, an indented `test:` block attaches to
the constructor or method before it.

Specialized tests place their modes before the optional name. `test` remains the
only test declaration.

```sev
test with property "reverse twice preserves values":
    values = [1, 2, 3]
    assert(values.reversed().reversed() == values)

test with bench "parser throughput":
    assert(parse("42") == 42)

test with chaos "read failures":
    assert(read() != absent)

test with property and chaos "generated input failures":
    assert(validate([1, 2, 3]))
```

The property runner controls case generation, random seeds, distributions, and
shrinking. The chaos runner derives a function's complete reachable failure
surface from the call graph, including failures introduced by callees. Tests add
returned values with `chaos.add(when function return value)` and thrown
exceptions with `chaos.add(when function throw error)`. This injection pattern
is valid inside any test and forbidden outside test scope. The runner injects
one event at a time by default, and handled events remain in the transitive
catalog. Compatible modes compose explicitly with `and`; commas do not combine
test modes.

## Imports

Quoted imports resolve project-relative source files; named imports resolve
through the package system. A quoted path may include or omit `.sev`.

```sev
import "helpers.sev"
import "local/geometry" as geometry
import math
import io as console

from math import dot
from io import print as write
```

Imports select names; they do not install packages. Package identity, versions,
sources, and dependency resolution belong to `package.toml`. A package uses the
standard Cargo-style layout:

```text
package.toml
sev.lock
src/lib.sev
src/main.sev
tests/
examples/
```

```toml
[package]
name = "geometry-app"
version = "0.1.0"
edition = "2026"

[compiler.type_resolution]
deny_any = true
deny_tensor_any = true
deny_unresolved = true
deny_inferred_fallback = true
deny_lost_type_information = true

[dependencies]
geometry = { version = "0.1.0", path = "../geometry" }
```

`[compiler.type_resolution]` is enforced after semantic checking and before MIR.
It rejects inference fallback, unresolved types or generics, lost type facts,
and implicit tensor erasure with `E000207`. Packages without these switches
remain flexible, and explicitly writing `Any` always records intentional dynamic
typing. See [`docs/error`](docs/error/README.md) for the categorized compiler
diagnostic catalog.

`from geometry import Point` resolves `geometry` from the current package, a
dependency selected by the manifest and lockfile, or the official library.
Official names are reserved and a declared dependency cannot shadow one. Source
files do not contain a separate `package` declaration.

The compiler looks for official packages in an explicit
`SEVERIAN_LIBRARY_PATH`, `$SEVERIAN_HOME/lib/severian/2026`, and an
executable-relative `../lib/severian/2026` installation before using its source
checkout or embedded distribution. Standard-library package sources, including
nested packages, are embedded in `sev`, so named imports remain available after
the compiler binary is installed or relocated. `SEVERIAN_LIBRARY_PATH`
deliberately selects editable packages for compiler and library development.

Native ABI declarations must acknowledge their unsafe boundary explicitly, and
only a declared library target may opt in:

```toml
[package]
name = "native-file-backend"
version = "0.1.0"

[package.unsafe]
capabilities = ["native-abi"]
sources = ["src/lib.sev"]

[lib]
path = "src/lib.sev"
```

```sev
unsafe:
    extern("__sev_file_read") def fileRead(path: string) -> Result[string, IOError]
```

Unsafe code is denied unless both its capability and exact source file appear in
`[package.unsafe]`. `native-abi` applies only to library targets; a binary cannot
use it even if listed. This prevents examples and applications from skipping an
API implementation with direct `extern(...)` declarations. Genuine low-level
examples can instead request a narrow capability such as `pointers` or
`runtime-owned-tasks` for one named source file. `test` bodies reject `unsafe:`
unconditionally. Application code imports the safe public package (`file` in
this case) instead of declaring platform symbols itself.

Public tensor APIs use `snake_case` names and rely on tensor types instead of
dtype-suffixed overload names where the operation is identical. For example,
`release[type](value: Tensor[type])` accepts `Tensor[bf16]`, `Tensor[f32]`, or
`Tensor[i64]` and infers `type` from each call; passing a non-tensor is a type
error. Backend-specific fixed-shape constructors keep their capacity in the
name when that capacity is part of the native ABI contract.

Registry packages use the same import syntax and never expose cache paths to
source code:

```toml
[dependencies]
tensor = "0.8"
http = "2.1"
local_geometry = { package = "geometry", path = "../geometry", version = "0.1" }
```

`SEVERIAN_REGISTRY` selects an on-disk registry (a path or `file://` URL), and
`SEVERIAN_HOME` selects Severian's local state directory (default `~/.sev`). A
registry stores immutable sources under `packages/<name>/<version>/` and trusted
SHA-256 digests under `checksums/<name>/<version>.sha256`. Resolution copies
verified sources to `~/.sev/packages/<name>/<version>/`, writes the exact
version, source, and checksum to `sev.lock`, and gives the compiler an
import-name-to-package map. Every later resolution verifies both registry and
cached source content; a modified cache is replaced before compilation.

`sev build`, `sev run`, `sev test`, and `sev check` resolve automatically.
`sev update` deliberately selects newer compatible releases; ordinary commands
honor existing lock selections. `sev add`, `sev remove`, and `sev publish`
manage the manifest, lockfile, and configured registry. Published versions are
immutable. `sev install <package>` builds a published binary into
`SEVERIAN_HOME/bin`. The repository's canonical manifest remains
`package.toml`; this is the Severian equivalent of Cargo's `Cargo.toml`.

Native/toolchain requirements are declarative as well. `[system]` contains
version requirements and `[install.<name>]` may select only a trusted vendor,
package identity, and `source = "vendor"`. Bare `sev install` shows and, after
confirmation, applies that plan; `--dry-run` only shows it and `--locked`
requires an exact existing lock. Publisher trust, domains, namespaces, dates,
and Ed25519 keys live under `SEVERIAN_HOME/trust`, outside package control.
`sev trust list`, `sev trust show <publisher>`, and `sev verify` expose the same
policy. Package installer scripts and executable hooks are rejected, and build
steps do not inherit install-time network or process authority. See the
[package guide](docs/examples/14-packages/README.md#external-requirements).

## Classes And Traits

Classes are value types by default. Traits describe capabilities, not inheritance
hierarchies.

```sev
trait Drawable:
    draw()

class Point: Drawable
    x: float
    y: float

    def Point(px: float, py: float):
        x = px
        y = py

    def draw():
        print(x, y)
```

Constructors are class-scoped functions with the same name as the class. A class
may define more than one constructor when the signatures are distinct.

```sev
class X:
    value: int

    def X(x: int, y: int):
        value = x + y

    def X(x: int):
        value = x
```

Inside a constructor, assigning to a declared field initializes that field on the
new instance. Methods and constructors access their current object's fields by
name without an explicit receiver parameter. `self` names the current execution
context, not a class instance.

Traits compose implicitly and transitively when one trait is named directly in
another trait's contract. There is no `extends` or `inherits` keyword.

```sev
trait Named:
    name() -> string

trait Drawable:
    Named
    draw()

class Button: Drawable
    label: string

    def name() -> string:
        return label

    def draw():
        print(label)
```

Implementing `Drawable` therefore satisfies both `Drawable` and `Named`.
Composition is dependency syntax: merely calling an operation declared by
another trait does not compose that trait.

The equivalent compact header form uses `+` only as a trait-list separator:

```sev
trait Visible:
    visible() -> bool

trait Drawable: Named + Visible
    draw()
```

Traits may also own semantic decorators and overlapping operator contracts.
Composition preserves the provider instead of turning the operation into a
global symbol.

```sev
trait XLA:
    @xla
    operator @(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]

trait Triton:
    @triton
    operator @(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]

trait Tensor: XLA + Triton
    @tensor(backend = auto)

@tensor(xla)
def multiply(left: Tensor[f32], right: Tensor[f32]) -> Tensor[f32]:
    return left @ right
```

Here the active context resolves `@` to `XLA::@`. Bare `@tensor` retains both
`XLA::@` and `Triton::@` as candidates; using `@` then reports `E000210` until a
provider is selected. A single candidate resolves automatically. Named
decorator arguments such as `backend = auto` are semantic policies, while
positional arguments such as `xla` are selectors.

Traits can also contribute structured behavior around an activated scope. Entry
and exit use ordinary Severian words rather than a separate hook language:

```sev
trait Time:
    @time
    with(context):
        context.timer.start()
    without(context):
        context.timer.stop()

trait Memory:
    @memory
    with(context):
        context.memory.begin()
    without(context):
        context.memory.end()

trait Profile: Time + Memory:
    @profile

def inference(value: Tensor[f32]) with { profile, memory < 4gb }:
    print(value)
```

The compiler enters `Time` and then `Memory`, runs the function, and removes
them in reverse order. Contract conditions and semantic behaviors can coexist
in one `with` set because every entry is type-resolved. A decorator is only
sugar for the trait-backed form:

```sev
@profile
def inference(value: Tensor[f32]):
    print(value)
```

Both forms produce the same HIR semantic context and structured scope. MIR
records reverse-order cleanup on normal fallthrough, return, and loop-control
exits; decorators never execute as arbitrary wrapper functions.

## Counts, Bytes, And Midpoints

`size(values)` returns the number of elements in a collection. `values.size()`
returns the number of bytes in the object. Severian does not provide `.len()`.

```sev
values = [10, 20, 30]

count = size(values)
bytes = values.size()
middle = values.mid()
```

`values.mid()` is the collection's midpoint primitive.

### Shape-Safety Hypothesis

Index-based iteration borrows the collection's shape for the loop. Safe code may
replace elements, but it cannot resize the collection while that shape is live.

```sev
for index in indices(values):
    values[index] += 1
```

Operations such as `pop`, `remove`, `clear`, and resizing are rejected inside
that loop. An opted-in low-level library may override the shape restriction in
an `unsafe` region, but indexing remains bounds-checked. Applications and tests
cannot make that override.

Frozen collections preserve both their contents and shape. Fixed arrays preserve
their shape while allowing element mutation. Resizable collections retain runtime
bounds checks whenever the compiler cannot prove an index belongs to their
current `indices(values)` set.

## Ownership

The compiler infers borrows, moves, and copies whenever it can. The reserved
prefix keywords `view`, `borrow`, `clone`, and `move` make an ownership operation
explicit when the program needs to say what it means. `view` creates a shared
read-only borrow, `borrow` creates an exclusive mutable borrow, `clone` creates
an independent owner, and `move` transfers ownership.

```sev
numbers := [1, 2, 3]

values_view = view numbers
print(values_view[0])

writable = borrow numbers
writable.push(4)

copy = clone numbers
owned = move copy
```

Parameter declarations contain names and optional types, not ownership modes.
An omitted type defaults to `Any` unless package type-resolution policy rejects
inference fallback; extern ABI parameters always require explicit types.
Parameters are viewed by default.
A call may use `view`, `borrow`, `clone`, or `move` on an
argument when the ownership operation must be explicit.

```sev
def update(values: list[int]):
    values.push(4)

update(borrow numbers)
```

## Optional Values

Optional values represent presence or absence without null. A function returning
`Option[type]` returns either `present(value)` or `absent`.

```sev
def find_name(id: int) -> Option[string]:
    if id == 1:
        return present("ada")

    return absent

switch find_name(1):
    present name:
        print(name)

    absent:
        print("missing")
```

## Errors

Recoverable errors are values. A fallible function returns a
`Result[type, exception]`, which contains either a successful value or a failure
exception.

```sev
def load(path: Path) -> Result[string, IOError]:
    data = read(path)
    return data
```

Assignment chooses how a `Result` is treated. `=` creates a stable binding and
`:=` creates a changeable binding; either one takes the successful value or
immediately propagates the failure from the current function. This keeps the
risk visible at the point where the value is taken instead of adding a second
error format to the function declaration.

Use `?=` to keep the complete `Result` without propagating it. The binding can
then be handled safely with `switch`:

```sev
outcome ?= read(path)

switch outcome:
    ok body:
        print(body)

    failure error:
        print(error)
```

A fallible expression can also be switched directly. `?=` requires a binding
name and a `Result` expression; it never unwraps or throws the failure.

Numeric parsing follows the same result flow for both integer and floating
point input:

```sev
count = int.parse("42")
ratio = float.parse("1.25")
```

Inside a function returning `Result[type, exception]`, returning a value of
`type` produces the successful result. Returning an expression that already has
the exact declared `Result` type forwards it unchanged. A bare `return` produces
a successful `unit` result when the declared success type is `unit`.

Severian accepts both its compact `switch` spelling and Python-compatible
`match`/`case` spelling for structural branching. Alternatives use `|` in
either form.

```sev
match extension:
    case ".yaml" | ".yml":
        print("yaml")
    case _:
        print("other")
```

## Function Contracts

A declaration introduces an API contract with `with`. This rule is identical
for functions and tests. The opening `{` and closing `}:` have their own lines,
each clause has its own line, and every clause ends in a comma. `sev fmt` writes
this canonical layout and `sev fmt --check` verifies it without changing files.

```sev
def run_job(job_id: int, connection: network.TCPConnection) with
{
    0 <= job_id <= 1000,
    connection != invalid,
    with connection,
}:
    process(job_id, connection)
```

The ordinary conditions are checked when the function is entered. A `defer`
condition is checked at entry and again only after an operation that can change
one of its dependencies:

```sev
def add(value: int, values: list[int]) with
{
    value >= 0,
    defer len(values) < 100 -> exception("list limit exceeded", location, vars),
}:
    values.append(value)
```

`exception` supplies the failure message. `location` adds a source location and
`vars` adds dependent names and values to the failure report.

A capability clause such as `with connection,` is compile-time metadata. A
caller must supply that capability explicitly with
`run_job(job_id, connection) with connection`; a missing or incorrect
capability is a compile-time error.

The capability belongs in the function contract and call suffix. Wrapping the
function's entire body in `with connection:` when the contract already requires
that capability is a compile-time error.

## Concurrency

Calls block by default. `async` starts work without blocking the current task and
returns a handle that can be joined with `await`.

```sev
worker = async fetch(url) with self
body = await worker
```

Channels use the PascalCase `Channel` class and an explicit `Buffer` policy.
Receiving is an ordinary `await` on the channel.

```sev
messages = Channel[string] with Buffer(16)
producer = async send "hello" with messages
message = await messages
```

Use `switch` when one task must receive from whichever of several channels is
ready. Exactly one ready arm commits; the other channel receives remain
untouched. The word after `from` names the source channel. An uppercase pattern
such as `Job from jobs:` destructures the received value and binds its declared
fields; a lowercase pattern such as `message from messages:` binds the entire
value under that name.

The optional `while` condition repeats selection without adding another indented
block. Its `with` setup runs once and remains scoped to the switch.

```sev
switch messages and commands while received < 2 with received := 0:
    command from commands:
        await handle(command) with runtime and lock
        received += 1

    message from messages:
        process(message)
        received += 1

    fail error:
        panic("Channels collapsed", error)
```

Every task names its lifetime owner. A task declared `with self` cannot outlive
the current execution. Runtime-owned task creation is confined to an opted-in
library; applications use its safe API rather than opening an unsafe region.

An imported execution package may add placement to the same clause. Local
distributed work keeps its structured owner and selects placement at the launch
site:

```sev
import distributed

with self and local:
    first = async process_shard(first_values)
    second = async process_shard(second_values)
    first_result = await first
    second_result = await second
```

The owner and placement are inherited by every bare `async` expression in the
block. `local` selects the native local-task backend and is retained on each
task spawn in MLIR. A task outside such a block must keep its explicit
`with self`/`with runtime` suffix. Placement is not a decorator: decorators
import domain syntax symbols and do not wrap a function or select where its
body executes.

Data-parallel regions use the same explicit `with` vocabulary, independently
of task ownership. A region may contain several kernels, or placement may be
attached to one `for` loop as compact syntax:

```sev
import parallel

with gpu:
    for index in indices(values):
        values[index] += 1

for index in indices(values) with simd:
    values[index] += 1
```

The two forms produce the same placement node in HIR. `gpu` selects GPU kernel
outlining for supported ranked `linalg` operations; `simd` records the request
for host-vector lowering. Existing shape-stability checks still reject a loop
that resizes the collection it is traversing.

Arguments passed to an async call are frozen by default. The child may read
them, but it cannot mutate the caller's values. Frozen arguments need no lock.
Code requests scoped access to a captured binding by naming it after the task
owner. The parent cannot perform a conflicting operation on that binding until
the child completes.

```sev
task = async do(x) with self and x
```

Here `x` remains owned by the surrounding scope, `self` owns the task, and the
borrow checker keeps the task's access to `x` within both lifetimes. Explicit
`clone x` and `move x` arguments remain available when the child needs an
independent value or permanent ownership transfer.

`with self and lock` transfers the lock capability to the child for the call.
The parent does not retain the lock while it waits. When several children need
the same mutable value, the lock serializes their access.

```sev
class Account:
    balance: int
    status: string

    def Account():
        balance = 0
        status = "surplus"

    def increment(amount: int):
        balance += amount
        status = "debt" if balance < 0 else "surplus"

    def decrement(amount: int):
        balance -= amount
        status = "debt" if balance < 0 else "surplus"

def main():
    account := Account()
    credit = async account.increment(10) with self and lock
    debit = async account.decrement(15) with self and lock

    await credit, debit
```

The lock protects the relationship between `balance` and `status`, not the
integer operations alone. Each child completes both field updates before the
other child may mutate the account. Calling either mutable method asynchronously
with only `with self` is rejected.

Use a lexical lock when several synchronous operations must form one exclusive
critical section:

```sev
with lock:
    increment(10)
    record_transaction("credit")
```

Mutable raw values otherwise cannot cross a task boundary. Frozen values permit
shared reads. Atomic values permit synchronized scalar mutation. Mutex locks
guard larger mutable state.

```sev
counter := atomic int 0
left = async counter += 1 with self
right = async counter += 1 with self

await left
await right
```

Tests and application binaries always use scope-owned work directly:

```sev
worker = async driver_call() with self
result = await worker
```

## Exponentiation

Exponentiation is native language syntax and is right-associative. Leading-zero
decimal notation is optional, so `.5` and `0.5` are equivalent.

```sev
square = value ** 2
square_root = value ** .5
power_tower = 2 ** 3 ** 2
```

Integer bases with non-negative integer exponents produce integers. Any float
operand produces a float. Integer overflow is checked, and a negative integer
exponent requires a float base.

## Math Mode

Most functions use ordinary expression syntax. A function can opt into reserved
domain symbols with decorators.

```sev
import tensor

@tensor(X)
def transform(a: Tensor[f64], b: Tensor[f64]) -> Tensor[f64]:
    return a X b
```

Decorator arguments name the symbols being imported into that function's syntax.
For example, `@tensor(X)` imports the tensor contraction meaning of `X`.

Outside decorated functions, those spellings are not silently reinterpreted. Each
decorator gives the compiler a link to the library or domain that owns the
symbols, their type rules, and their lowering behavior.

The decorator package must first be imported unless the decorator is declared
by a local semantic trait. Decorators are retained as typed compiler metadata;
they are not runtime Python-style function wrappers. Trait-owned decorators
additionally retain the active traits, provider-qualified operator and
operation candidates, scoped `with`/`without` behavior, and named policies.

Integer bit operations use the `bits` capability. They resolve automatically
from integer operands, or a decorator can isolate an explicit symbolic subset.

```sev
import bits

@bits(|, &, ^)
def combine(left: int, right: int) -> int:
    return (left | right) ^ (left & right)
```

The `bits.Bits[T]` trait declares those operator contracts. Default Boolean
algebra remains the short-circuiting language syntax `and`, `or`, and `not`, so
ordinary Boolean expressions need neither an import nor a decorator.

The same idea can reserve words for non-math domains.

```sev
import regex

@regex(match)
def has_slug(text: string) -> bool:
    return match text with r"[a-z]+-[0-9]+"
```

## Fixtures

The examples in `docs/examples` are source fixtures. As the parser and driver are
implemented, every fixture should move from "documented syntax" to "compiled by
tests".

The folders are ordered so the compiler can grow in passes:

1. `00-getting-started` through `03-collections-iteration` cover the Python-like
   core: indentation, bindings, calls, control flow, and built-in collections.
2. `04-classes-traits` through `07-generics-constraints` introduce Rust-flavored
   structure: value classes, traits, ownership, results, patterns, and generic
   constraints.
3. `08-concurrency` through `10-numerics-mlir` layer in Go-style concurrency,
   systems boundaries, and MLIR-oriented numeric kernels.
4. `12-enums-aliases` onward cover evolving features: enums, method
   mutation contracts, Cargo-like packaging, specialized tests, and
   compiler-stage fixture organization.
