# SIP-0027: Ownership over MIR Execution

Status: Draft — WIP skeleton
Type: Compiler
Created: 2026-10-03
Target: `sev_compiler/mir/ownership`

## Context

Ownership consumes [MIR](0026-MIR.md) and its verified [CFG](0028-CFG.md). It checks initialization, moves, loans, escapes and cleanup across executable paths while retaining the same HIR blocks, definitions and source locations.

Compilation grouping (`partition_owner`) and lexical containment do not establish who owns a value. Ownership facts come from declared type/callable effects and resolved operations.

## Data model

| Component | Contract |
| --- | --- |
| Existing `Place`, `Projection`, `LocalDecl`, `ValueId` | Identify storage, subobjects and values within a callable realization. Do not replace them with a second expression model. |
| Existing `OwnershipPlan` | Currently records views, storage, aliases and origins; evolve this contract rather than adding another ownership summary with overlapping meaning. |
| Proposed flow facts | Initialization/move state and active loans at operation points and CFG edges, keyed by realization and place/value. |
| Proposed loan record | Borrowed place, access mode, provenance and validity region. Aliasing must account for projections and edge arguments. |
| Proposed cleanup actions | Drop/release obligations at specific exits, with source origins and ordering; distinguish planned actions from actions already inserted. |

Facts and plans reference the analyzed body revision. They do not own copied blocks or acquire module membership independently. Callable effect summaries use resolved identities so recursive calls can be checked without executing source functions.

## Interfaces and flow

Proposed contracts, in pseudocode:

```text
analyze_ownership(analysis: MirAnalysis, callable: realization)
    -> OwnershipPlan | diagnostics
apply_cleanup(analysis: MirAnalysis, plan: OwnershipPlan)
    -> success | diagnostics
verify_ownership(analysis: MirAnalysis, callable: realization)
    -> success | diagnostics
```

Analysis reads the CFG, operation effects and callable/type contracts. It propagates facts to a fixed point across branches and loops, including supported exceptional exits. A use requires validity on every incoming executable path; uncertainty cannot silently become permission.

Cleanup insertion is an explicit mutation pass. It preserves effect order and provenance, invalidates affected analyses, and is followed by verification. Repeating preparation must not insert duplicate cleanup. Recursive effect summaries require a fixed point or an explicit unresolved-contract diagnostic.

## Problems to avoid

- Do not infer copyability, destruction or borrowing from primitive names or a central literal enum; consume type-owned contracts.
- Check partial moves, overlapping projections and escaping aliases, not only whole local variables.
- Preserve both the originating operation and conflicting use in diagnostics. Avoid success-shaped fallback plans after errors.
- Unsupported control effects must stop readiness until their ownership behavior is defined.

## Testing

Use real MIR output to test a move on one branch followed by a joined use, a loan across a loop, and an escaping reference to local storage. Check cleanup exactly once on normal/early/exceptional exits and rejection of stale plans after CFG changes.

## Open work

Define the state lattice, projection-overlap rules, loan lifetimes, exceptional cleanup and interprocedural summaries. Replace the ownership package's legacy HIR module imports with the shared program contract.
