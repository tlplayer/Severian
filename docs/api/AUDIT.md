# API completeness and relationship audit

This audit records whether the specification layers agree. It complements the
[weakness ledger](WEAKNESSES.md): the ledger tracks known missing capability;
this page tracks unclear or mismatched relationships between documented APIs.

## Layer relationships

| Relationship | Required invariant | Current evidence | Status |
| --- | --- | --- | --- |
| Source syntax → Universal | A source spelling maps to a structural identity; dtype, rank, and shape are operands or type data rather than symbol suffixes. | `index.toml` cross-checks syntax families against Universal enums and structural tensor IDs. | Enforced for registered primitives, operators, literals, conversions, AST families, and tensor operation classes. |
| Universal → HIR/MIR/LIR | Type substitutions remain distinct from runtime tensor specialization and no library function is selected by source-name matching. | Generic and tensor records name their structural lowering contracts. | Partial; dependency generic bodies and all tensor paths are tracked in `W-GEN-001` and `W-TENSOR-001`. |
| LIR declaration → MLIR library | An unresolved ABI symbol has one registered owner; only declarations may be replaced; imported MLIR and the combined module both verify. | `core.text.string` v1 registry and composition tests. | Implemented for String concat, compare, and release. |
| MLIR library → platform ABI | Severian behavior is represented by MLIR operations. External declarations may call platform allocation or OS services but do not hide Severian semantics in C. | `library/core/text/mlir/string_v1.mlir`. | Implemented for the initial String slice. |
| Source String → versioned String ABI | Ordinary source values use `{data, length, capacity}` and borrowed views use `{data, length}`. | `StringAbiV1` and `StringViewAbiV1` exist in the runtime contract. | Transitional: ordinary lowering still uses the NUL-terminated pointer compatibility representation. |
| StorageView → tensor specialization | Storage metadata determines rank, dimensions, strides, element representation, and target before rank-dependent MLIR/TTIR emission. | Runtime specialization and launcher records plus runtime tests. | Implemented for the tested launcher slice; complete operation/device coverage remains open. |
| API record → `.sev` evidence | Every implemented record names tests and every primitive folder contains a standalone Severian conformance program. | Native `sev api check` scans records, source exports, topology, and snippets. | Enforced by the catalogue checker. |
| Editor grammar → language/API catalogue | Highlighted keywords and primitive types agree with the specification index. | `[editor]` groups in `index.toml` are the declared editor vocabulary. | Syntax coverage exists; semantic editor integration remains `W-IDE-001`. |

## String migration boundary

The String MLIR work deliberately exposes two contracts instead of conflating
them:

1. `__sev_string_concat`, `__sev_string_compare`, and
   `__sev_string_release` are demand-loaded MLIR compatibility exports for the
   current pointer-valued source representation.
2. `StringAbiV1` is the intended owned boundary, but ordinary source lowering
   does not consume it yet.

Therefore it is correct to claim that those three operations are no longer
implemented in C. It is not yet correct to claim that source Strings support
embedded NUL bytes or that all remaining String runtime behavior has moved out
of `string.c`.

## Audit rules

- A status is based on executable evidence, not the intended architecture.
- A compatibility adapter must name the representation it preserves and its
  removal condition.
- A new library operation must identify its structural owner, ABI version,
  effects, and external platform dependencies.
- C-compatible linkage is permitted at a foreign/platform boundary; a C body is
  not accepted as a new Severian library implementation.
- A backend limitation must not change the source or Universal operation
  identity.
