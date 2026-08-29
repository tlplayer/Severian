# Library API index

The library catalogue is source-bound to every `library/**/package.toml`.
Package presence and stable API identity are exhaustive; five mature surfaces
also validate every exported top-level Severian declaration against
`export_sources`.

| Group | Stable IDs | Relationship |
| --- | --- | --- |
| [Compute](compute/README.md) | `library.compute*`, `library.tensor` | Execution/fusion policies over typed values. |
| [Core](core/README.md) | `library.core*`, `library.collections` | Prelude-adjacent values, memory, text, time, math, and protocols. |
| [Data](data/README.md) | `library.data*` | Parsing, querying, and storage formats. |
| [Harness](harness/README.md) | `library.harness` | Executable test/benchmark orchestration. |
| [Interop](interop/README.md) | `library.interop*` | ABI and FFI boundaries. |
| [Media](media/README.md) | `library.media*` | Audio, graphics, and plotting. |
| [Model](model/README.md) | `library.model*` | Model architecture and codec compositions. |
| [Network](network/README.md) | `library.network*` | Transport, HTTP, and TLS. |
| [System](system/README.md) | `library.system*`, `library.file`, `library.process` | OS-visible resources and orchestration. |
| [Tensor](tensor/README.md) | `library.tensor`, `tensor.*` | Source compositions over structural tensor operations. |
| [Testing](testing/README.md) | `library.testing` | User-facing testing helpers. |

Package IDs name import surfaces. They do not imply that every source file is
public or that a backend implements every operation. Per-symbol contracts and
backend matrices refine package presence; they never replace it.
