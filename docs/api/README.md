# Severian API specification

This first-class specification lives under `docs/api/`; examples elsewhere in
the repository demonstrate usage but do not independently define the API.

`docs/api/` is the normative specification of Severian's public surface. It
answers **what exists, what it means, and what is promised**. Focused Markdown
pages are the human contract; TOML records are the machine-checkable index.
`docs/examples/` answers **how larger programs use the features together**.

An API record is not an aspiration disguised as documentation. Every record
has an explicit status, evidence, tests, and limitations. A feature may be
specified before it is implemented, and an implementation may be partial on a
specific backend, but those facts must be visible here.

## Contract levels

1. **Language API** — syntax, types, operators, generics, effects, ownership,
   errors, control flow, concurrency, testing, and unsafe boundaries.
2. **Prelude API** — names available without a package import.
3. **Library API** — exported package symbols and their source contracts.
4. **Compiler API** — structural operations, compiler hooks, artifact
   boundaries, target capabilities, and runtime specialization records.

The initial inventory concentrates on the type/operator/generic/tensor spine:

- [Primitives](primitives/README.md)
- [Generics and shape parameters](generics/README.md)
- [Operators](operators/README.md)
- [Tensor API and structural operations](tensor/README.md)
- [Python/Rust symmetry methodology](SYMMETRY.md)
- [Generic notation](APPENDIX.md)
- [Weakness ledger](WEAKNESSES.md)

It is intentionally incomplete; the weakness ledger records gaps without
pretending they are supported.

Every primitive registered by Universal owns a folder under
[`primitives/`](primitives/README.md). Each folder contains a detailed contract
and standalone Severian conformance program; the checker derives the expected
folder names directly from `PRIMITIVES`.

## Record rules

Machine records contain one or more `[[feature]]` tables conforming to
[`schema/feature.schema.json`](schema/feature.schema.json). Stable IDs use
dotted lowercase names such as `operator.add` or `tensor.matmul`.

Each Markdown page owns the explanation, examples, edge cases, and lowering
rationale for one coherent API section. It links back to stable feature IDs;
TOML does not replace those explanations.

The required fields are:

| Field | Contract |
| --- | --- |
| `id` | Globally stable feature identity. |
| `kind` | Feature category, independent of implementation. |
| `syntax` | Source spelling or structural notation. |
| `type_params` | Generic metavariables defined in the appendix. |
| `parameters` | Ordered value/operand contract. |
| `constraints` | Type, shape, value, effect, or ownership constraints. |
| `returns` | Result contract; `unit` for no value. |
| `effects` | Observable mutation, allocation, I/O, async, or unsafe effects. |
| `errors` | Statically or dynamically produced errors. |
| `ownership` | Copy, move, borrow, alias, or mutation behavior. |
| `universal` | Universal/HIR structural identity. |
| `lowering` | Lowering boundary, never a dtype/rank-specific symbol. |
| `status` | `specified`, `implemented`, `partial`, `experimental`, `deprecated`, or `unavailable`. |
| `since` | First language edition/version carrying this contract. |
| `tests` | Conformance evidence. Implemented records require it. |
| `examples` | User-facing examples. |
| `limitations` | Known unsupported or incomplete behavior. |

## Validation

Run:

```bash
sev run docs/api --bin api-check
sev run docs/api --bin api-symmetry
```

The Severian checker validates records, unique IDs, statuses, referenced
snippets, implemented-feature evidence, and the editor vocabulary. It extracts
every Markdown `sev` fence and runs a sequential, CPU-pinned, memory-bounded
compiler check. The symmetry runner compares Severian output with small Python
or Rust reference programs under the same bounded execution policy. The
intended driver interface is:

```text
sev api list
sev api show operator.add
sev api check
sev api diff
```

Until those commands exist, `check.sev` and `symmetry.sev` are the native
self-hosted conformance tools and are explicitly listed as such.

See [APPENDIX.md](APPENDIX.md) for generic notation and
[SYMMETRY.md](SYMMETRY.md) for behavioral comparison rules, and
[WEAKNESSES.md](WEAKNESSES.md) for the current capability gaps.
