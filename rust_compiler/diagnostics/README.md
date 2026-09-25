# Compiler diagnostic contract

A source failure must retain its structured diagnostic until the reporting
boundary. A valid report contains:

- A diagnostic code and a concrete explanation of the failure.
- The owning package and compiler stage.
- A primary source span and the exact file snapshot used to produce that span.
- Every related declaration or import, with its own span and explanation.
- Actionable help describing what the user can change.

The text format is code/cause, package, stage, primary file/line/column and
excerpt, related locations, notes, then help. Source IDs alone are insufficient:
each referenced ID must identify exactly one attached file snapshot. Spans must
be ordered, in bounds, and on valid character boundaries in the source model.

Use `Diagnostic::source_error` for a complete source report. Infrastructure
errors without a source location use `Diagnostic::operation_error`; their
location is explicitly reported as not applicable. The Severian equivalents
live in `sev_compiler/syntax/diagnostics/diagnostics/src/lib.sev`.

`Diagnostic::new` remains a partial builder for existing parser/semantic code.
Before turning its result into text, attach sources and package/stage context;
module-graph callers use `ModuleGraph::contextualize`. Import indexing, import
planning, and semantic analysis do this at their public boundaries, including
failures raised before full compilation starts.

`Diagnostic::validate` checks the contract, including related and nested errors.
Rendering always validates. An incomplete legacy report retains its original
cause and any available source IDs/offsets, and emits an explicit
`diagnostic contract: incomplete diagnostic` explanation. It must never silently
drop a location, choose between conflicting source snapshots, or replace the
original failure with a secondary panic. Legacy producers that trigger this
message still need migration; the renderer does not invent missing context.

Regression tests are inline. Run without building the compiler executable:

```sh
cargo test -p severian-diagnostics
cargo test -p severian-semantic scope_resolution_tests
```
