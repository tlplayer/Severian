# XXI

XXI connects Severian values to FFI contracts and ABI representations. Ownership
remains the compiler ownership library's authority, initialized contents belong
to `core.storage`, and allocations belong to `core.memory`.

For example, a C provider can receive an initialized byte view without knowing
Severian's list representation:

```sev
@c(symbol = "consume")
def consume(data: borrowed[list[u8]]) -> int

@c(symbol = "fill")
def fill(data: inout[list[u8]]) -> int
```

XXI generates a bridge from these contracts. The C view is defined in
[`extern/c/bytes.h`](extern/c/bytes.h): a pointer and an initialized byte length.
Borrowed input crosses by value; `inout` receives a pointer to the view. Inout
may modify contents but must preserve the address and initialized length.

Each bridge retains the caller's storage owner for the full call and copies the
initialized bytes into temporary `core.storage` storage. The provider sees no
Severian list header. Inout copies the validated contents back; borrowed input
does not modify the caller. Cleanup releases both the temporary storage and the
loan on the original owner. Overlapping mutable arguments are rejected before
invoking the provider. Allocations therefore appear in the existing storage
statistics, without a separate allocator or ownership registry.

FFI plans retain the original value contract, parameter mode and result contract
alongside the ABI representation. The driver retains their ownership/lifetime
modifiers in HIR and registers transfers and borrowed results by resolved
function identity in the existing type context consumed by MIR ownership.
`out`/`inout` lowering no longer loses its original contract when selecting a
pointer representation.

The currently executable generated bridge supports borrowed and inout
`list[u8]`, scalar arguments/results, and a nonvariadic target C ABI. Unsupported
element types, implicit sequence ownership, and returned foreign sequences are
rejected. A returned foreign allocation needs an explicit lifetime and release
protocol; it must not be silently adopted as Severian storage. The compiler's
other existing scalar boundaries remain available.

Source-compiler buffer calls use [`src/buffer.sev`](src/buffer.sev) for region and
returned-count validation over typed caller-owned memrefs. It obtains addresses
through `core.memory` without replacing the compiler's buffer ownership plan.
The Rust seed generates sequence bridges; the native source frontend still has
its existing limitations loading the public `@file` dispatch package.

`budget`, `step`, and `account` bound calls and cumulative transferred bytes.
`region` checks initialized bounds without overflowing offset arithmetic;
`transfer` checks returned counts and optionally rejects zero progress; `count`
and `byte` validate narrowing conversions before performing them. IO uses these
operations and ordinary XXI declarations, with no IO-specific semantic lowering.

These checks run before and after an in-process foreign call. They cannot
interrupt code that never returns. For a hard deadline, use
[`worker.run`](src/worker.sev) with an isolated provider executable, a time limit,
and a memory limit. It delegates termination and resource limits to
`system.process`. Violations of a C view's address/length contract terminate the
calling process; use the worker boundary when that failure must be isolated.

Validation covers generated bridge execution, owner retention and cleanup,
invalid output lengths, numeric overflow, bounded progress, and a provider that
loops forever. `test/validation/performance/libraries.sh` writes the measurements
and logs beneath `package.pkg/debug/profile/libraries-*/`.
