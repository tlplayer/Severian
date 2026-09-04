# Agent IR

Agent IR is Severian's compiler-derived semantic interface for development
tools and software agents. It exposes the compiler's existing understanding of
a package; it does not parse source again and is not an MLIR dialect.

```text
source -> lexer -> parser -> AST -> semantic/HIR -> MIR -> Agent IR
                                                   -> MLIR
```

Emit it for one source target or package binary with:

```sh
sev build --emit agent-ir path/to/package
sev build --emit agent-ir --bin compiler path/to/package
sev build --emit agent-ir -o /tmp/compiler-ir path/to/compiler.sev
```

Without `-o`, the directory is `target/agent-ir` beneath the selected input.
The output is deterministic for an unchanged semantic graph.

## Version 1 layout

```text
target/agent-ir/
├── package.json
├── symbols.jsonl
├── declarations.jsonl
├── types.jsonl
├── tests.jsonl
├── diagnostics.jsonl
├── source-map.json
└── graphs/
    ├── calls.json
    ├── dependencies.json
    ├── ownership.json
    ├── references.json
    └── types.json
```

`package.json` declares `"agent_ir": 1`, entrypoints, stable module IDs, the
root module's source-defined API, and record counts. JSONL keeps declarations,
symbols, types, tests, and diagnostics independently streamable. Graph files
store explicit `from`, `relationship`, and `to` edges.

IDs start with the existing compiler-term abbreviation where one applies:

- `T:` Type, `V:` Value, `E:` Error, `Ex:` Expression
- `M:` Macro, `S:` Statement, `D:` Declaration, `P:` Pattern, `L:` Literal
- `O:` Operation, `I:` Instruction, `B:` Block
- `A:` Argument, `R:` Result, `F:` Callable
- `C:` Constraint, `K:` Kind, `Y:` Symbol, `N:` Node, `X:` any compiler term

IDs are opaque stable anchors. Human-readable names and package-relative source
paths are separate fields, so moving a checkout does not rewrite graph
identity. Function declarations include signature, source span, effects,
throws, and four hashes:

- `source_hash`: exact source module contents
- `semantic_hash`: normalized HIR/MIR meaning without source-location metadata
- `interface_hash`: callable boundary and dispatch contract
- `dependency_hash`: outgoing semantic graph targets

Tests are retained during emission and connected to their compiled callable.
`diagnostics.jsonl` is empty after a successful build; structured diagnostics
for failed partial graphs are a later format extension, not fabricated from
terminal output.

Agent IR is read-only in version 1. Impact queries and hash-guarded semantic
patching can build on these stable IDs without making this format a second
frontend or type system.
