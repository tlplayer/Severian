# Severian library

`library/` is the source tree for Severian's official library. The directory is
not part of an import path: programs write `import network` or
`from math import sqrt`, not `import library.network`.

The organization borrows three useful ideas without copying any one ecosystem:

- Python's broad, task-oriented coverage and searchable category index.
- Rust's small foundation, explicit prelude, and separation between portable
  abstractions and platform services.
- Go's short, focused, flat package names.

## Ownership boundary

Every public operation has one implementation owner:

| Owner | Responsibility | Examples |
| --- | --- | --- |
| compiler | Language primitives, type checking, ownership, and intrinsics | `int`, `string`, `Result`, borrowing, `size` |
| library | Public APIs and portable Severian algorithms | `boolean`, `probability`, JSON values |
| runtime | OS calls and native machinery used by library APIs | sockets, files, clocks, entropy |

The compiler must not silently invent a package API. Runtime-backed packages
need typed Severian declarations and a runtime symbol mapping before they are
considered implemented. Their initial packages are therefore marked
`interface-pending` rather than filled with placeholder functions.

## Package shape

Each package is independently testable and documented:

```text
library/math/
├── Severian.toml
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
3. Put algorithms in Severian source when practical; use the runtime only for
   capabilities that require the OS, native code, or a compiler intrinsic.
4. A runtime-backed API must have a typed public declaration, tests, and a
   documented failure model before it is stable.
5. Security-sensitive implementations such as cryptography and TLS must wrap
   reviewed native providers; they must never begin as toy implementations.
6. Package tests belong beside the package and are written in Severian. Rust
   tests may verify the compiler, but are not the package's primary test suite.

Run all currently implemented library packages with:

```sh
tools/check_library.sh
```

