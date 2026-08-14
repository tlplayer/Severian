# Severian diagnostics contract

Severian diagnostics are source-level explanations, not stringified compiler
failures. Terminal output, JSON output, editor integrations, and `sev explain`
must all be projections of the same structured diagnostic.

## Stable error namespace

Errors use `E` followed by exactly six decimal digits. Severian does not use a
separate runtime prefix: compilation and execution share one searchable
namespace.

| Range | Responsibility |
| --- | --- |
| `E0001xx` | lexing and syntax |
| `E0002xx` | names, calls, types, traits, and contracts |
| `E0003xx` | ownership and borrowing |
| `E0004xx` | memory and bounds safety |
| `E0005xx` | arithmetic safety |
| `E0006xx` | concurrency and task safety |
| `E0007xx` | unsafe capabilities and foreign boundaries |
| `E0008xx` | recoverable results and effects |
| `E0009xx` | runtime failures |
| `E0024xx` | tensors, shapes, and accelerator operations |
| `E0099xx` | compiler failures awaiting a narrower translation |

Codes are never reused for a different meaning. A broad code may be replaced
by a new, narrower code, but existing documentation remains a stable reference.
Warnings and naming guidance retain their existing `W` and `N` namespaces.

## User-facing order

Every specific error should provide, in this order:

1. severity, stable code, and a concise problem statement;
2. the primary source location and the smallest useful highlighted expression;
3. secondary labels showing where a requirement or conflicting state began;
4. notes explaining inferred types, runtime values, or the causal chain;
5. one or more concrete fixes when the compiler can construct them;
6. `sev explain EXXXXXX` when deeper documentation exists.

The first screen should answer “what failed, where, why, and what can I do?”
Compiler implementation terms belong only in `diagnostics = "internal"` output.

## Labels and causal chains

A diagnostic has one primary label and may have secondary labels in the same or
other files. The primary label identifies the rejected operation. Secondary
labels identify origins: a parameter declaration, trait requirement, previous
move, active view, inferred generic binding, or incompatible tensor axis.

Labels are preferable to prose containing line numbers because editors can
navigate them and source changes do not make the explanation stale.

## Suggestions and applicability

A suggestion contains a message, an applicability, and one or more text edits.

- `machine-applicable` means the compiler knows the complete syntactically and
  semantically valid replacement. Editors may offer one-click application.
- `maybe-incorrect` means the edit is a useful preview but requires judgment,
  such as choosing a meaningful argument value or parsing untrusted text.

The terminal renderer previews single-line edits. JSON preserves every edit and
its byte, line, and column range so editor clients do not need to parse terminal
text. Suggestions must never silently change program meaning merely to satisfy
the type checker.

## Backend translation boundary

MLIR, StableHLO, LLVM, XLA, PJRT, linker, and native signal text are internal
evidence. Before a backend operation is emitted, its Severian HIR/MIR operation
must retain source provenance and relevant type or shape facts. A translation
layer then maps a backend failure to a Severian diagnostic.

For example, a `stablehlo.dot_general` contracting-dimension failure becomes
`E002401`, labels the Severian matrix multiplication, prints both operand
shapes, highlights the incompatible axes, and explains the required equality.
The raw verifier record is included only in internal diagnostics. If translation
is not yet possible, `E009900` is used instead of exposing an unstructured
backend message.

## Explanation pages and tests

`sev explain` pages describe the invariant, common causes, valid repairs, and a
small example. They do not merely repeat the one-line error.

`sev errors` enumerates the complete six-digit E-code catalog, turning the
compiler's safety surface into a reviewable checklist.

Each stable error has a source fixture under `docs/error/`. Catalog tests require
the filename's code to be the first emitted code, a precise location and source
snippet, and a registered explanation. Representative diagnostics additionally
have golden assertions for labels, notes, edit previews, applicability, and JSON
fields.

## Current implementation sequence

The initial implementation covers the shared structured model and renderer,
six-digit E codes, parser token insertion, type-boundary explanations, missing
and misspelled argument fixes, mandatory initialization, enum exhaustiveness,
compile-time zero-divisor checks, runtime source ranges, automatic native crash
stacks, structured JSON edits, `sev errors`, and richer explanations. The next
diagnostic families should add structural object diffs, option-flow proofs,
trait requirement origins, ownership history, and complete tensor/backend
translation without changing this protocol.
