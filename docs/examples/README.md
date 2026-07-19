# Example Fixtures

This directory contains complete `.sev` snippets used by the language docs and
future compiler tests.

The folder order follows the shape of the official Go, Rust, and Python learning
paths while staying Severian-first:

- Go contributes the habit of small runnable programs, packages, generics, and
  direct concurrency.
- Rust contributes ownership, borrowing, traits, pattern switching, results, and
  fearless concurrency.
- Python contributes readable indentation, rich collections, modules, classes,
  and a gentle flow from simple expressions to larger programs.

## Progression

| Folder | Purpose |
| --- | --- |
| `00-getting-started` | Smallest runnable programs and imports. |
| `01-values-control` | Literals, bindings, operators, conditionals, loops. |
| `02-functions-modules` | Function signatures, defaults, imports, modules. |
| `03-collections-iteration` | Lists, tuples, maps, sets, ranges, iteration. |
| `04-classes-traits` | Value classes, methods, traits, trait composition. |
| `05-ownership-borrowing` | Inferred ownership plus explicit view/borrow/clone/move keywords. |
| `06-results-patterns` | `Result`, `Option`, `?=`, `present`, `absent`, and exhaustive switching. |
| `07-generics-constraints` | Type parameters and trait-bounded abstractions. |
| `08-concurrency` | `async`, `await`, bounded channels, channel switches, tasks, and safe shared state shapes. |
| `09-systems-unsafe` | Pointers, unsafe blocks, and isolated low-level code. |
| `10-numerics-mlir` | Tensor-style code that should lower cleanly to MLIR. |
| `11-testing` | Function-attached and constructor-attached tests. |
| `12-enums-aliases` | Placeholder enum and type alias syntax. |
| `13-method-mutation` | Placeholder method mutation contracts. |
| `14-packages` | Cargo-like official package layout and manifest. |
| `15-tests` | Ordinary, property, benchmark, chaos, and composed tests. |
| `16-compiler-stages` | Placeholder parser/semantic/ownership/lowering fixture stages. |
| `17-servers` | TCP request/response, channel-based chat, and map/reduce services. |
| `bugs` | Invalid-and-fixed safety contracts for future diagnostic tests. |

For now these are syntax fixtures that define the language target. Once the
compiler driver exists, each file should be compiled by automated tests, starting
with parser acceptance and then advancing to semantic checking and MLIR lowering.

Suggested fixture pipeline:

```sh
tools/check_docs_examples.sh
```

By default the script runs `sev check` for every valid `.sev` file in this
directory, and verifies that every `bugs/**/invalid.sev` fixture fails at its
documented source location.
Set `SEV=/path/to/sev` to use a locally built compiler driver.
