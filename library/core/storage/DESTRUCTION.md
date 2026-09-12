# Owned destruction contract

Status: partially implemented in the source compiler; general compiler and
storage implementation remains incomplete.

The baseline is Rust's [destructor and drop-scope model](https://doc.rust-lang.org/stable/reference/destructors.html):
custom cleanup precedes recursive owned cleanup, initialization determines what
is destroyed, and moves transfer responsibility. Severian currently uses reverse
field order as specified below; it does not copy every Rust temporary-scope rule.

Memory owns allocation and physical release. Storage owns initialized elements
and the allocation that contains them. Collections manage their elements,
length, and capacity through storage. Class cleanup uses those same ownership
operations.

## Class customization

The class cleanup hook has a consuming receiver and a non-throwing unit result:

```sev
class Connection:
    handle: Handle
    pending: buffer[Message]

    operator drop(move self) -> unit:
        disconnect(handle)
```

The source compiler accepts this hook syntax; `Connection` is an illustrative API.
`disconnect` represents class-specific cleanup; field and allocation destruction
remain automatic after the hook. A class that does not define a hook gets the
same generated field/storage destruction without the custom body.

The compiler-generated destruction operation must:

1. Consume the owning value exactly once, after outstanding borrows have ended.
2. Run the class's custom cleanup body, if present, while its fields are valid.
3. Destroy remaining initialized, owned fields in reverse declaration order.
4. Release the owning storage through its memory provider after its elements
   and fields have been destroyed.

The hook does not recursively invoke its own destruction. It cannot return or
resurrect `self`, let a borrow of `self` escape, or throw through cleanup.
Successfully moved-out or explicitly destroyed fields are not destroyed again.
Borrowed fields do not own the referenced allocation. A value stored inline has
no separate allocation to free merely because its class has a cleanup hook.

## Default destruction

Default destruction is derived from semantic ownership and the physical
storage contract. It is not a blind walk that frees every pointer field.

| Value | Default action |
| --- | --- |
| Scalar or non-owning view | End its lifetime; release no referenced allocation |
| Owned string or buffer | Destroy owned contents as required, then release owned storage |
| Collection | Destroy initialized owned elements, then release its backing storage |
| Class without a hook | Destroy its remaining initialized owned fields |
| Class with a hook | Run the hook, then perform the same default field destruction |
| Enum or union | Destroy only the active payload and its owned storage |
| Moved or uninitialized place | Perform no destruction |

Storage capacity does not imply initialized elements. Partial construction,
element removal, and moved fields must update the initialized-place state used
by generated cleanup.

## Compiler obligations

Ownership analysis must carry moves, loans, initialization, and escape facts
into destruction lowering. Cleanup is required on ordinary scope exit, early
return, error propagation, reassignment, and loop exits. Reassignment must not
lose the previous owner; returning or moving a value transfers its cleanup
responsibility to the new owner.

Both compiler implementations must use the same contract. Source-owned
memory/storage operations remain the release boundary; a class hook must not
introduce a separate allocator or bypass that boundary.

Existing `operator drop()` hooks and working storage release paths remain until the
replacement syntax, dispatch, and generated cleanup are implemented and tested.
Adding parser support alone does not implement this contract.

## Required validation

Native allocation and destructor counters must verify exactly-once cleanup for
default and custom destruction, nested owned fields, initialized collection
elements, active sum payloads, partial construction, moves, reassignment,
returns, errors, and loop exits. Tests must check hook-before-fields and reverse
field order, as well as reject double destruction, use after move, escaping
borrows, and invalid destructor results/effects.

Compiler memory profiles must also demonstrate reclamation of ordinary
temporary strings, lists, and boxed values. Passing existing explicit-resource
tests does not establish cleanup for that object graph.

## Implemented checkpoint (2026-09-12)

`tests/sev_compiler/destruction.py` exercises native source-compiler behavior:
custom hooks followed by nested fields in reverse order, default record cleanup,
direct partial moves, explicit field destruction inside a hook, early returns,
returned records and optional payloads, reassignment, and loop-local cleanup on
continue and break. Existing `def drop()` methods use the same generated cleanup.
Explicit throws clean local resources before propagating their saved error.
Consuming parameters belong to the whole function scope.

Returning a moved field preserves sibling cleanup. A hook's `self` cannot be
moved as a whole. Partial movement out of a class with a custom hook is rejected
outside its hook. Conditional resource/field destruction must agree across both
continuing branches; nested partial resource paths are rejected until full place
initialization tracking exists.

Remaining implementation includes arbitrary owned collection elements and
partially constructed storage, general error-propagation edges, dynamic
initialized-place flags, and cleanup of the bootstrap compiler's string/list/
boxed-object graph. Existing MLIR buffer destruction remains the physical release
path for supported source-owned values. Hook-order tests alone do not prove
complete storage reclamation.

The Rust bootstrap rejects `operator drop` with an explicit diagnostic until
class destruction lowering exists there. It now reclaims provably nonescaping
string concatenation intermediates after their last borrowing operation. That
conservative optimization does not establish general destruction.
