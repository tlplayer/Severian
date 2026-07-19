# Severian

Severian is a compiled systems language with Python-like syntax, Rust-like safety,
and MLIR as the compiler backbone.

The repository is being built piece by piece. The current focus is the language
front end:

- `compiler/ast`: source-level syntax tree nodes.
- `docs/language`: living language notes.
- `docs/examples`: example `.sev` programs that should become compiler fixtures.
- `docs/examples/14-packages`: placeholder Cargo-like package layout.

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

## Example Fixtures

Every source snippet in the language docs should have a matching file under
`docs/examples`. Once the parser and driver exist, those files should be compiled
as part of the test suite.
