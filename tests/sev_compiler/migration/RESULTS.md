# Migration validation — 2026-09-08

The six-gate migration is **not complete**. The unchanged-binary runner passes
16 of its 30 cases. Gate 3's five syntax cases all pass, together with the
semantic-helper mutation and generic implementation cases. The executable
still lowers structured MIR; it does not expose the required canonical CFG
Agent IR boundary.

The working-tree changes are based on commit
`6c1c28d3ef299fefaaceee2a8a90f1f478b32dab`. The tested source compiler's SHA-256
is `04928daea9882e5c904c1064eb8f594f697bbdc79fc690acf835a63a2dbddc6d`.
Rebuilds may produce a different executable hash; the source-only tests verify
that their selected executable stays unchanged during each case.

| Gate | Passing cases | Remaining acceptance work |
| --- | --- | --- |
| 1. Baseline | 4/5 | Inspection of the active canonical CFG pipeline |
| 2. Definitions | 2/5 | Inheritance inspection, absence serialization/validation, resolved compiler capability enforcement |
| 3. Source syntax | 5/5 | Acceptance floor passes; broader qualification and definition-system work remains in the inventory |
| 4. Semantic execution | 3/5 | Indexed place evaluation, lazy operands and user truth; compiler-semantic CFG execution remains unfinished |
| 5. Canonical CFG | 1/5 | Source branch construction, provenance, terminator verification and typed errors |
| 6. Library and retirement | 1/5 | Actual primitive libraries, collection protocols, definition removal through IR provenance, and physical retirement of alternatives |

Other validation on the final implementation:

| Command | Result |
| --- | --- |
| `(cd sev_compiler && ../target/debug/sev build)` | Pass |
| `python3 tests/sev_compiler/source_contracts.py` | 7/7 pass |
| `SEVERIAN_SKIP_BUILD=1 python3 tests/sev_compiler/bootstrap_mlir.py` | 117 checks pass |
| `SEVERIAN_SKIP_BUILD=1 python3 tests/sev_compiler/callable_bodies.py` | 31 cases pass |
| `SEVERIAN_SKIP_BUILD=1 python3 tests/sev_compiler/diagnostics.py` | 6 cases pass |
| `target/debug/sev test tests/sev_compiler/semantic_ir` | 9/9 pass |
| `git diff --check` | Pass |

`[G:Add]` now uses ordinary parameter/constraint parsing. Semantic registration
resolves the constraint's definition and inherited grammar context, and the
parameter can be renamed. Source-contract regressions cover `[Syntax:Fuse]`,
inherited semantic bodies, mismatched bindings/signatures, unresolved
requirements, and an unfamiliar symbol implemented by both a primitive and a
user-defined record through one generic function. These tests compile and run
natively without rebuilding the source compiler.

See [INVENTORY.md](INVENTORY.md) for the permanent compiler boundary and the
remaining implementation/removal ledger. These results do not establish full
language migration or canonical-IR/native differential execution.
