# Agent IR

Agent IR is Severian's compiler-derived semantic interface for development
tools and software agents. It exposes the compiler's existing understanding of
a package; it does not parse source again and is not an MLIR dialect.

Install the compiler from the current checkout before inspecting work that has
not reached a release:

```sh
./install.sh --source
hash -r
sev --help | grep agent-ir
```

The final command prevents an older `sev` with the same version number from
silently remaining first on `PATH`.

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

## Version 2 layout

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

`package.json` declares `"agent_ir": 2`, entrypoints, stable module IDs, the
root module's source-defined API, and record counts. JSONL keeps declarations,
symbols, types, tests, and diagnostics independently streamable. Graph files
store explicit `from`, `relationship`, and `to` edges.

IDs use the [canonical compiler-term vocabulary](../../sev_compiler/universal/README.md):

- `T:` Type, `V:` Value (including arguments), `E:` Error, `Ex:` Expression
- `M:` Macro → operations, `O:` Operation (including statements/instructions)
- `L:` Literal, `B:` Block, `R:` Result, `F:` Callable
- `W:` With-clause operations (including constraints), `C:` Container
- `S:` Shape, `N:` Number of elements, `Y:` Symbol, `G:` Grammar
- `X:` Any compiler term, including declarations and modules
- `To:` Token, `Lx:` Lexeme

The vocabulary does not imply that every category has an emitted record today.
Module IDs use `X:module:…`; class declarations use `X:declaration:…`; trait
constraints use `W:…`. Argument records have `kind: "V"`, `role: "argument"`,
and IDs containing `:argument:` to distinguish them from other values.
Descriptive `source:` and `test:` prefixes are artifact roles, not additional
compiler-term abbreviations.

Version 2 changes the module, argument, declaration and constraint ID namespaces.
Consumers must rebuild stored indexes when moving from version 1; old IDs are
not emitted as compatibility aliases. The source compiler's separate CFG JSON
schema is unchanged by this directory-format revision.

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

Agent IR is read-only in version 2. Impact queries and hash-guarded semantic
patching can build on these stable IDs without making this format a second
frontend or type system.
