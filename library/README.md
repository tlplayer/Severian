# Severian library

`library/` is the source tree for Severian's official library. The directory is
not part of an import path. `import math` makes the package's available names
callable directly, while `from math import sqrt` selects only `sqrt`. Programs
never write `import library.math`.

```sev
import math

dot([1.0, 2.0], [3.0, 4.0])
```

Package functions are ordinary `def` declarations with real bodies. There is
no separate export header or `extern def` form.

The initial general-purpose surface is available through flat imports such as
`core`, `list`, `set`, `string`, `math`, `random`, `file`, `path`,
`json`, `regex`, `time`, `process`, `environment`, `http`, `network`, `tls`,
and `tensor`. Collection APIs are typed and mutating operations act
on the supplied collection; OS-backed behavior is implemented behind the
trusted `platform` boundary.

Visualization is split into two layers: `graphics` owns explicit rendering
targets and drawing primitives, while `plot` builds charts from lists, `Data`,
and tensors without global figure state. The portable reference backend is
headless SVG; window, event, and GPU backends remain behind the same public
render-target model.

Length and storage use distinct compiler-level vocabulary: `len(value)` and
`size(value)` are exact aliases for the number of elements, while
`bytes(value)` and `bits(value)` report storage size. `capacity(value)` reports
allocated element slots for resizable containers. The same operations are
available as methods. For a statically shaped `Tensor[f32, 32, 128]`, both
`len()` and `size()` are 4096, `bytes()` is 16384, and `bits()` is 131072.

The organization borrows three useful ideas without copying any one ecosystem:

- Python's broad, task-oriented coverage and searchable category index.
- Rust's small foundation, explicit prelude, and separation between portable
  abstractions and platform services.
- Go's focused, flat package organization, while keeping ordinary words complete.

## Ownership boundary

Every public operation has one implementation owner:

| Owner | Responsibility | Examples |
| --- | --- | --- |
| compiler | Language primitives, type checking, ownership, and intrinsics | `int`, `string`, `Result`, borrowing, `size` |
| library | Public APIs and portable Severian algorithms | `boolean`, JSON values |
| platform | typed native ABI used underneath library APIs | sockets, files, clocks, entropy |

The compiler must not silently invent a package API. Native-backed packages use
typed Severian `native(\"symbol\")` declarations inside explicit `unsafe:` blocks in the `platform`
package; the explicit marker acknowledges the host ABI boundary.
There is no implicit source-level native namespace. A package is considered
implemented only when its native test executable links, runs, and matches its
expected stdout and stderr.

## Package shape

Each package is independently testable and documented:

```text
library/math/
├── package.toml
├── README.md
├── src/
│   └── lib.sev
└── tests/
```

The package manifest is the source of its name, edition, implementation owner,
and stability. Public imports remain flat even when [CATALOG.md](CATALOG.md)
groups packages by subject.

## Design rules

1. Keep the automatic prelude small: primitives and universally required
   control/result types only.
2. Prefer one obvious package for a concept. Do not create both `net` and
   `network`, or `fs` and `file`.
3. Put algorithms in Severian source when practical; use `platform` only for
   capabilities that require the OS, native code, or a compiler intrinsic.
4. A platform-backed API must have a typed public declaration, tests, and a
   documented failure model before it is stable.
5. Security-sensitive implementations such as cryptography and TLS must wrap
   reviewed native providers; they must never begin as toy implementations.
6. Package tests belong beside the package and are written in Severian. Rust
   tests may verify the compiler, but are not the package's primary test suite.

## Numerical stack

The experimental numerical stack separates author-facing machine-learning code
from lowering policy. `tensor` is the one canonical numerical container, while
`model` and `model.neuralnet` provide model and layer APIs such as `Linear`,
`LayerNorm`, `MultiheadAttention`, and `TransformerEncoderLayer`. Historical
top-level `models`, `matrix`, and `neuralnet` package names are not part of the
public hierarchy. `parallel` owns operation-local
`simd`, `simt`, and `gpu` contracts used inside libraries. Compatible activation
chains are fused automatically; model callers do not request fusion or select a
backend.

## Data and infrastructure stack

`pql` validates typed schemas and query structure before emitting SQL or running
deterministic fixtures. `storage` provides provider-neutral relational,
document, key/value, Dynamo-style, and future object-storage plans; it does not
execute operations. `database` owns real SQL connections, transactions,
iterable rows, persistence, and a TCP database server. `vm`, `container`, and `hypervisor`
describe validated host plans above the explicit `platform` ABI, while
`orchestrator` supplies desired-state scheduling and reconciliation. Privileged
host mutation remains a distinct executor boundary and is never performed by
ordinary package tests.

## Operating-system laboratory

`kernel` provides hosted, deterministic OS policies for physical pages, virtual
mappings, capabilities, processes, files, syscalls, interrupts, and scheduling.
The executable under `docs/lab/operating_system` composes them with
native channels and tasks. It is intentionally distinct from a freestanding,
bootable target, whose missing ABI and architecture requirements are documented
beside the lab.

Run all currently implemented library packages with:

```sh
tools/check_library.sh
```
