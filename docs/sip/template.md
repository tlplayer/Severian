SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

Summary

Describe the proposal in a few paragraphs.

What changes?

What is the intended end state?

Motivation

Describe the concrete problem this SIP solves.

Include:

user-facing limitations;

compiler/runtime/tooling limitations;

duplicated or conflicting implementations;

architectural boundaries being violated;

performance, safety, maintainability, or portability problems.

Avoid proposing implementation details here unless they are necessary to explain the problem.

Goals

State what this SIP must accomplish.







Non-goals

State what this SIP intentionally does not solve.







Current state

Describe how Severian implements this today.

Current flow

input
  ↓
current component
  ↓
current representation
  ↓
output

Components involved

Component

Current responsibility

Problem

Lexer





Parser





Semantic





HIR





MIR





MLIR





Runtime





ABI





Package system





Tooling





Current files / APIs

path/to/current/file.sev
path/to/current/module/

Current invariants

List behavior that must remain true during migration.







Problems with the current design

List specific problems.

1. Problem

Current behavior:

...

Why it is a problem:

...

2. Problem

...

Proposed design

Describe the desired end state.

Proposed flow

input
  ↓
new source of truth
  ↓
canonical representation
  ↓
consumers

Core rules







Syntax

If applicable:

# proposed Severian syntax

API

If applicable:

# proposed Severian API

Types / IR

If applicable:

# proposed type or IR representation

Ownership and safety

Describe any effects on:

ownership;

lifetimes;

mutation;

concurrency;

unsafe operations;

runtime checks.

Errors and diagnostics

Show expected diagnostics for invalid use.

ErrorType: explanation

Source of truth

State which component owns the semantics after this SIP.

Source of truth:
    ...

Consumers:
    ...

No consumer should independently recreate these semantics unless explicitly justified.

Compiler impact

Fill only applicable sections.

Lexer

Changes:

Removed behavior:

New tests:

Parser

Changes:

Removed behavior:

New tests:

Semantic

Changes:

Removed behavior:

New tests:

HIR

Changes:

Removed behavior:

New tests:

MIR

Changes:

Removed behavior:

New tests:

MLIR / backend

Changes:

Removed behavior:

New tests:

Runtime

Changes:

Removed behavior:

New tests:

ABI

Changes:

Removed behavior:

Compatibility concerns:

Package system

Changes:

Removed behavior:

Compatibility concerns:

Tooling

CLI:

LSP/editor:

debugger:

formatter:

profiler:

documentation:

Compatibility

Describe effects on existing Severian code.

Source compatibility

Compatible:

Breaking:

Automatically migratable:

Binary / ABI compatibility

Compatible:

Breaking:

Version boundary required:

Package compatibility

.pkg:

.pkgi:

manifest:

lockfile:

Migration plan

The migration must have a finite path from the current design to the proposed design.

Stage 0 — Define behavior

Add tests that describe the desired end state before changing implementation.

Required:

positive compile tests;

negative compile tests;

end-to-end behavior;

regression coverage for current valid behavior.

Stage 1 — Introduce the new path

Implement the minimum new architecture alongside the current path.

Do not migrate all consumers yet.

Exit criteria:

new behavior passes isolated tests;

no existing behavior regresses.

Stage 2 — Migrate internal consumers

Move compiler/runtime/tooling consumers to the new source of truth.

Track each consumer:

Consumer

Old path

New path

Status









Exit criteria:

all intended consumers use the new path;

no new code is allowed to depend on the deprecated path.

Stage 3 — Migrate examples and public APIs

Update:

docs/examples/;

compiler bootstrap code;

standard library;

tests;

package APIs;

editor/tooling integrations.

Exit criteria:

examples demonstrate only the preferred design;

compatibility behavior is isolated.

Stage 4 — Deprecate old behavior

Mark old APIs, syntax, passes, representations, or files as deprecated.

Every deprecation must identify:

deprecated item
replacement
migration path
removal condition

Warnings must be actionable where user-facing behavior is involved.

Stage 5 — Delete compatibility path

Delete the old architecture after all known consumers have migrated.

Exit criteria:

deprecated code removed;

compatibility adapters removed;

duplicate tests removed or rewritten;

dead configuration removed;

obsolete documentation removed.

Deprecation and cleanup plan

This section is required.

A SIP is not complete only because the new implementation works.

It is complete when obsolete architecture has either been removed or explicitly justified as permanent.

Deprecated concepts

Item

Replacement

Deprecated in

Removal condition









Files to delete

path/to/obsolete/file.sev
path/to/obsolete/module/

For each file, state why the file becomes unnecessary.

APIs to delete

old_api(...)
OldType
OldTrait

Passes / compiler flows to delete

old lowering path
duplicate resolver
compatibility rewrite

Configuration to delete

old.manifest.option
OLD_ENVIRONMENT_VARIABLE
--legacy-flag

Tests to delete or rewrite

path/to/legacy_test.sev

Do not retain tests whose only purpose is preserving behavior this SIP intentionally removes.

Documentation to delete or rewrite

docs/old-design.md
docs/examples/old-example/

Temporary compatibility code

Any compatibility layer introduced during migration must have an owner and deletion condition.

Compatibility layer

Needed for

Delete when







Cleanup verification

Before marking the SIP implemented:

Search for references to deprecated APIs.

Search for references to removed types and symbols.

Search for legacy compiler paths.

Search for compatibility flags.

Remove dead files and modules.

Remove duplicate tests.

Remove obsolete documentation.

Remove stale diagnostics.

Remove unused configuration.

Confirm no alternate source of truth remains.

Run dead-code and unused-symbol checks.

Run the full regression suite.

Tests

Tests are part of the proposal, not an implementation afterthought.

Unit

...

Compile-positive

test:
    ...

Compile-negative

test with compiler: reject:
    ...

Expected diagnostic:

...

End-to-end

source
  ↓
compiler
  ↓
runtime/backend
  ↓
expected result

Regression

Identify existing behavior that must remain valid.

Migration

Add tests proving both:

the new path works;

the legacy path disappears when Stage 5 completes.

Performance

If performance-sensitive, specify:

Metric

Baseline

Required

Compile time





Runtime





Memory





Binary size





End-to-end example

Provide at least one complete .sev example demonstrating the final design.

# complete example

Expected:

...

Implementation plan

Break implementation into pieces small enough to land independently.

Change 1

Scope:

...

Tests:

...

Change 2

Scope:

...

Tests:

...

Change 3

Scope:

...

Tests:

...

Each change should leave the tree buildable and testable.

Alternatives considered

Alternative 1

Description:

Why rejected:

Alternative 2

Description:

Why rejected:

Risks

Risk

Impact

Mitigation







Open questions







Open questions should be resolved before Accepted unless they are explicitly deferred.

Completion criteria

A SIP may move to Implemented only when:

Proposed behavior is implemented.

Positive tests pass.

Negative tests pass.

End-to-end tests pass.

Internal consumers use the new source of truth.

Examples use the new design.

Public documentation is updated.

Migration is complete.

Deprecations have reached their stated removal condition.

Obsolete files, APIs, passes, and compatibility layers are deleted.

No unintended duplicate implementation remains.

Performance requirements are satisfied, if applicable.

Full regression suite passes.

If legacy behavior remains intentionally, document it here with the reason it is permanent.

Decision record

YYYY-MM-DD — Draft

Initial proposal.

YYYY-MM-DD — Accepted

Decision and rationale.

YYYY-MM-DD — Implemented

Implementation commits:

<commit>
<commit>

Cleanup commits:

<commit>
<commit>

Final architecture

When implemented, replace this section with the final architecture rather than leaving only the proposal.

final source of truth
       ↓
final consumers
       ↓
final output

Removed

old subsystem
old compatibility path
old API

Retained intentionally

component — reason