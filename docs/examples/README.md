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
| `15-tests` | Ordinary, property, benchmark, chaos, integration, and composed tests. |
| `16-compiler-stages` | Placeholder parser/semantic/ownership/lowering fixture stages. |
| `17-servers` | TCP request/response, channel-based chat, and map/reduce services. |
| `bugs` | Invalid-and-fixed safety contracts for future diagnostic tests. |

For now these are syntax fixtures that define the language target. Once the
compiler driver exists, each file should be compiled by automated tests, starting
with parser acceptance and then advancing to semantic checking and MLIR lowering.

Compile every example in lexical order and then run its attached tests:

```sh
tools/check_docs_examples.sh
```

The script aggregates failures instead of stopping at the first one. It also
verifies that every `bugs/**/invalid.sev` fixture fails with the diagnostic in
its adjacent `expected-error.txt` file. Add `--native` to compile and execute
programs containing `main()`, compare them with adjacent `.stdout` files, and
publish only verified executables under `bin/examples`. Set
`SEV=/path/to/sev` to use a specific
compiler; otherwise the script builds and uses `target/debug/sev`. It exits with
a nonzero status when any compilation, test, native build, or expected
diagnostic fails.

Ordinary tests remain fast and controlled. Integration tests opt into the real
native executable and its system boundaries:

```sh
sev test example.sev --integration
sev test example.sev --integration-only
```

Inside `test with integration`, `stdout` and `stderr` contain the captured native
process output and can be checked with ordinary Severian assertions.
