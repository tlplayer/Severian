# Severian

Severian is a compiled systems language with Python-like syntax, Rust-like safety,
and MLIR as the compiler backbone.

The repository is being built piece by piece. The current focus is the language
front end:

- `compiler/ast`: source-level syntax tree nodes.
- `library`: official Severian packages, manifests, documentation, and
  language-native tests.
- `runtime`: native services used by runtime-backed library packages.
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

## First Compiler Slice

The compiler currently accepts the first fixture, `01-hello.sev`, through the
complete lexer, parser, AST, semantic, HIR, ownership, lowering, and driver
pipeline.

```sh
cargo run -p severian-driver --bin sev -- check docs/examples/00-getting-started/01-hello.sev
cargo run -p severian-driver --bin sev -- compile docs/examples/00-getting-started/01-hello.sev
cargo run -p severian-driver --bin sev -- run docs/examples/00-getting-started/01-hello.sev
```

`compile` verifies the emitted MLIR, translates its LLVM dialect to LLVM IR, and
links a native executable named `a.out` by default. Use `-o executable` to choose
another path. `emit-mlir` prints the intermediate MLIR for inspection, while
`run` executes the validated HIR for a fast development loop.

## Example Fixtures

Every source snippet in the language docs should have a matching file under
`docs/examples`. Once the parser and driver exist, those files should be compiled
as part of the test suite.

Run the ordered compile-and-test harness with:

```sh
tools/check_docs_examples.sh
```

Add `--native` to also invoke the MLIR/LLVM native compiler for every accepted
example containing `main()`. Successful executables mirror the source tree under
`bin/examples`.

## Official library

The official library uses flat imports such as `import network` and
`from math import square`. Its package catalog and compiler/library/runtime
ownership boundary are documented in `library/README.md` and
`library/CATALOG.md`.

Run every library package that currently has an implementation with:

```sh
tools/check_library.sh
```
