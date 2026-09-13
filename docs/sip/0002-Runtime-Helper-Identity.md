# SIP-0002: Typed identities for generated runtime calls

Status: Implemented (working-tree candidate for review; not committed)
Type: Compiler
Authors: Severian contributors, Codex
Created: 2026-09-10
Target: `rust_compiler`; regression contracts for `sev_compiler`
Supersedes: None

## Summary

A generated helper's semantic identity must include its ordered parameter types
and result type, in addition to its external symbol. Keep the external symbol
unchanged for linking. Reuse a declaration only when its complete typed signature
matches. Apply this rule uniformly instead of special-casing aggregate helpers.
At the native boundary, explicitly convert integer collection operands and results
to and from the C helper's `int64_t` transport type.

This is a bounded fix for a reproducible Rust compiler failure encountered while
extending the source compiler. It does not resolve the entire memory-profile
audit, or establish parity for all function, ownership, and test modes.

## Problem and reproduction

At baseline commit `5188e5e0`, this valid Rust-compiled program fails MIR verification:

```sev
integers: list[list[int]] = [[1]]
strings: list[list[string]] = [["hello"]]
print(integers[0][0])
print(strings[0][0])
```

Run `sev_rust build nested.sev --emit mir -o nested.mir`. The observed error is:

```text
MIR pass verify failed: call argument type mismatch:
TypeId(218922139) is not assignable to TypeId(252467705)
```

The error also prints the callee's numeric `DefId`. The meaningful mismatch is
between `list[string]` and `list[int]`, the elements of the two outer lists.
Deleting either outer list hides the collision. Source declaration order can
therefore determine which signature subsequent calls inherit.

The original [memory audit](../../sev_compiler/package.pkg/examples/memory-profile-current/REPORT.md)
reported 62/111 builds and 46 passing test runs. Those are historical measurements,
not results of this SIP. This reproduction is independently checked against the
current baseline; it does not depend on the original audit logs.

## Current flow and cause

```text
list[list[int]] literal    -> append(storage, list[int])
list[list[string]] literal -> append(storage, list[string])
                                     |
                                     v
                    same C symbol: __sev_list_append_list
                                     |
                                     v
             ensure_runtime_function caches by symbol alone
                                     |
                                     v
                second call reuses first typed declaration
                                     |
                                     v
                       MIR rejects the argument type
```

[`ensure_runtime_function`](../../rust_compiler/frontend/semantic/src/lib.rs)
already includes parameter and result types in the identity of `_aggregate`
helpers. Other helpers use only the symbol. The same problem applies to return
types: indexing different nested lists uses `__sev_list_index_list` with different
semantic results. Fixing append alone would leave that defect.

[`verify`](../../rust_compiler/transforms/mir/src/verify.rs) correctly checks calls
against typed declarations. Weakening assignability would hide the producer's
mistake. The C runtime intentionally shares implementations across compatible
representations; it is not the authority for source type identity.

## Goals and non-goals

- Preserve exact parameter and result contracts in HIR and MIR.
- Deduplicate identical typed helpers and distinguish different signatures.
- Keep runtime symbols and native layouts unchanged.
- Keep mutation classification and pointer-storage provenance working.
- Test both semantic verification and executable behavior.

This SIP does not add nested owned buffers to the source compiler, change borrow
or move semantics, fix unrelated advanced test modes, redesign all FFI handling,
or make numeric type IDs into a cross-version persistent identity format.

## Proposed design

Use a structured registry key containing `symbol`, `parameters: Vec<TypeId>`, and
`result: TypeId`. Generate the synthetic `DefId` from all three using the existing
aggregate signature encoding. Store the original symbol separately in the key
and in `ExternalCall.symbol`.

```text
typed expression
  -> key(symbol, ordered parameter types, result type)
  -> exact typed declaration / DefId
  -> verified MIR call
  -> existing representation and ABI lowering
  -> original external C symbol
```

Core invariants:

1. Equal keys return the same definition without adding another declaration.
2. Different parameters or results yield different semantic identities, even
   when their native representation and symbol are shared.
3. Symbol classification reads the symbol field, not the signature encoding.
   This matters for exact `__sev_list_clear` / `__sev_list_address` checks and
   pointer helpers recognized by a `_slot_u8` suffix.
4. MIR retains its existing argument and result verification.
5. ABI lowering still owns representation conversion and external declaration
   emission. Distinct semantic identities do not authorize incompatible native
   declarations for one symbol.

The registry currently creates only native-runtime calls with a fixed interface,
FFI, ABI, and `universal_boundary` policy. If those become variable, extend the
key with the variable contracts before sharing entries. Do not infer ownership
policy from equal machine layouts.

### Native integer transport

The expanded native test exposed a second defect after the identity fix:
`list[i32]` and `list[i64]` reached MLIR with different declarations of
`__sev_list_append_i64`. MLIR rejected an `i64` argument against the first `i32`
declaration. This is why passing HIR or MIR verification alone is insufficient.

[`CfgLowering`](../../rust_compiler/transforms/lowering/src/lib.rs) now routes
native-runtime list/set helpers ending in `_i64` through explicit runtime calls,
as it already does for aggregate helpers. Integer operands of at most 64 bits are
extended to the C `int64_t` slot representation; signedness controls extension.
Integer results return as 64 bits and are converted back to their semantic width.
Values already 64 bits retain their bit pattern. Pointers, booleans, and aggregate
arguments retain their existing representations. External user functions are
unaffected because this route requires the `native-runtime` interface.

These are internal storage conversions after semantic verification, not newly
permitted source conversions. HIR/MIR signatures remain concrete. Normalized LIR
runtime calls let the existing emitter declare each C symbol with a consistent
native signature. Tests cover negative `i8`, maximum `u16`, all-set `u64` bits,
indexed writes, and both mixed-width declaration orders. This does not redefine
unsigned collection ordering or arithmetic overflow.

## Applying the lesson to the source compiler

The source compiler uses source generic specialization and typed buffer operations;
it has no equivalent symbol-only `runtime_definitions` registry to patch.
Its [`definitions.sev`](../../sev_compiler/frontend/semantic/src/definitions.sev)
deliberately rejects buffers containing owned values:

```text
E000205: buffer elements require copyable storage;
owned elements require destruction lowering
```

The reproduction reaches this diagnostic when compiled with `sev_compiler`.
Do not remove this safety boundary merely to make a Rust regression pass there.
Add source regression coverage for multiple supported typed buffers in one
program, both declaration orders, and continued rejection of nested owned
buffers. Existing raw-memory, explicit-ownership, and case-table tests remain
acceptance gates. Keep `TestCaseRow` for now: removing it is a separate source
model change, and passing a Rust test does not prove source buffer destruction.

## Regression matrix

| Case | Expected contract | Regression caught |
| --- | --- | --- |
| `list[list[int]]` plus `list[list[string]]` | Rust builds and prints `1`, `hello` | Parameter signature collision |
| Reverse declaration order | Same output when printed in the same order | First-registration dependency |
| Index both outer lists | Each result keeps its element type | Result-only identity collision |
| Repeat the same typed operations | One declaration per typed signature | Unbounded duplicate declarations |
| Aggregate collections | Existing behavior preserved | Removal of aggregate special case |
| Multiple supported primitive buffers | Both compilers preserve values | Shared representation erases source types |
| Narrow signed/unsigned and full-width bit patterns | Correct extension and recovery | ABI fix silently corrupts values |
| Clear list, then inspect it | Mutation remains visible | Classifier accidentally reads encoded signature |
| Address a buffer element; write through pointer | Original storage changes | Loss of slot-pointer provenance |
| Append an incompatible value | Semantic rejection | Accidental weakening of type checking |
| Nested owned buffer in source compiler | Existing destruction diagnostic | Unsupported ownership silently accepted |

These tests guard against future regressions; they do not predict that a future
revision will necessarily fail. Baseline and post-change observations are recorded
below rather than marking unsupported cases as passing or skipping them.

## Implementation and cleanup

1. Add failing semantic regression tests and capture the baseline failure.
2. Replace the string key with the typed key; migrate both symbol classifiers in
   the same change. Remove the `_aggregate`-only identity branch.
3. Normalize integer collection transport in MIR-to-LIR lowering. Keep the typed
   signatures and conversion boundary independently reviewable.
4. Add native Rust tests and source acceptance/rejection tests. Check relevant
   existing ownership, pointer, aggregate, and test-table regressions.
5. Rebuild the source compiler with the candidate Rust compiler as a bootstrap
   gate. Record any blocker explicitly; do not call this implemented prematurely.

No syntax, public API, runtime ABI, or compatibility alias is deprecated. The
symbol-only cache and special-case branch are removed immediately, with no
dual-write registry or fallback. Existing aggregate tests remain useful; no test
is deleted to obtain a pass. Completion requires the new regression tests to pass,
the old architecture to be absent, and no new failures in the comparison suites.
Pre-existing failures must be reproduced against the saved baseline and recorded.

## Validation and rollback

Use the candidate Cargo-built binary explicitly: `sev_rust` prefers the release
binary and can otherwise accidentally test an older build. Preserve the baseline
binary outside the build directory before rebuilding. Run from the repository root:

```sh
RUST_MIN_STACK=16777216 cargo test -p severian-semantic -p severian-mir -p severian-lowering
cargo test -p severian-driver --test method_bodies --test conversions
cargo build --release -p severian-driver --bin sev
package.pkg/release/sev build sev_compiler --bin sev_compiler -o /tmp/sev-source-sip-0002
export SEVERIAN_SOURCE_COMPILER=/tmp/sev-source-sip-0002
python3 tests/sev_compiler/runtime_helper_contracts.py
python3 tests/sev_compiler/raw_memory.py
python3 tests/sev_compiler/explicit_ownership.py
python3 tests/sev_compiler/test_cases.py
```

The native tests must link and run, not merely emit HIR. Source tests use the
already-built source compiler and assert its binary hash does not change during
the test. Keep original audit results immutable; any full rerun goes into a new
output directory and reports build, executable, and test outcomes separately.
The larger Rust test-thread stack avoids a stack overflow observed in the existing
`builders_ordered_constraints_and_mocks_lower_to_executable_control_flow` test;
the same suite passed with 16 MiB. This setting changes the test harness, not the
compiler's ownership rules.

Rollback triggers include native ABI errors, changed output, pointer corruption,
new ownership failures, or a bootstrap regression. Restore the saved compiler
binary for immediate use. Revert only this SIP's implementation changes (or its
dedicated commits in reverse dependency order once committed), then rebuild with
the restored seed and rerun the previous passing gates. Preserve unrelated edits;
do not reset the repository. Keep the reproduction and mark the SIP Draft again
with the failing evidence. Use fresh output paths after rollback so candidate
objects are not mistaken for baseline artifacts.

For this working-tree candidate, baseline binaries and a reverse-applicable patch
are saved in `/tmp/sev-sip-runtime`. Before rolling back, review the patch and
check that no later edits overlap it:

```sh
git apply --reverse --check /tmp/sev-sip-runtime/implementation.patch
git apply --reverse /tmp/sev-sip-runtime/implementation.patch
cp /tmp/sev-sip-runtime/sev-rust-before package.pkg/release/sev
cp /tmp/sev-sip-runtime/sev-source-before sev_compiler/package.pkg/host/dev/bin/sev_compiler
```

Then rebuild the debug Rust binary before using Cargo integration tests; otherwise
it may still contain the candidate. The patch covers the three modified Rust files,
including their new tests. The SIP and new source regression file are retained.
Save these temporary artifacts elsewhere before `/tmp` cleanup if rollback must
remain available. No rollback has been applied.

## Risks and alternatives

Including more signatures increases declaration counts, bounded by distinct
typed signatures actually requested. It can expose previously hidden ABI
inconsistencies, which is why native execution is mandatory. Synthetic helper
IDs change; rebuild compiler outputs rather than mixing old and new artifacts.

Special-casing `_list` as well as `_aggregate` would fix only today's example.
Canonicalizing everything to a C representation during semantic analysis would
discard the types MIR is supposed to verify. Weakening MIR checking or retaining
only wrapper-record workarounds would conceal the underlying collision.

## Decision and validation record

Proposal first described and implemented on 2026-09-10, starting from `5188e5e0`.
The implementation is uncommitted and available for review.

| Validation | Result |
| --- | --- |
| Baseline nested-list CLI reproduction | Failed MIR verification |
| Baseline new semantic regression | Failed: one helper declaration instead of two |
| Rust semantic / MIR / lowering unit suites | 140 / 6 / 3 passed |
| Rust native method / conversion integration suites | 18 / 5 passed |
| Candidate release builds `sev_compiler` | Passed |
| New source runtime-helper contract suite | 3 passed |
| Source raw-memory suite | 6 passed |
| Source explicit-ownership suite | 15 passed |
| Source case-table suite | 3 passed, 2 failed on both saved baseline and candidate |
| Original 111-example audit | Not rerun; original report retained |

The two source case-table failures are unchanged and remain open:

1. `03-testing/02-with-tests/10-parameterized.sev` rejects
   `assert(a in [1, 2, 3])` with `E000212: operator result does not match expected type`.
2. `test_expect_and_later_case_failures_are_not_suppressed` gets a failing process
   exit as expected, but stderr says `Aborted (core dumped)` without the required
   assertion diagnostic. The diagnostic assertion has not been weakened.

These need separate source fixes for membership contextual typing and assertion
diagnostic emission. They are not caused by runtime-helper identity and are not
counted as successful tests. The broader audit still needs work.

The tested source binary and the source binary rebuilt with the final Rust code
have identical SHA-256:
`3d0577da69fea0b2dde78677bdff0baa5cab7929002e1b776a8367feae3156a1`.
Logs and baseline binaries for this run are under `/tmp/sev-sip-runtime`.

Regression implementations:
[`semantic tests`](../../rust_compiler/frontend/semantic/src/lib.rs),
[`native tests`](../../rust_compiler/boundaries/driver/tests/method_bodies.rs),
[`source contracts`](../../tests/sev_compiler/runtime_helper_contracts.py).
