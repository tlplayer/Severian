# Ownership, storage, and memory

`ownership` is the [compiler library](../../../sev_compiler/frontend/ownership/README.md) authority for lifetimes. `storage` owns initialized
contents. `memory` supplies physical allocation and release. These are distinct
responsibilities with one direction of dependency:

```
ownership library contracts -> checked ownership plan -> MLIR ownership -> lowering
                                      |
                                  core.storage
                                      |
                                   core.memory
```

Ordinary arguments are views of caller-owned values. Reading a view does not
transfer or release its owner. Exclusive access permits checked mutation; an
explicit consuming boundary transfers ownership. A returned owned value must
acquire or transfer its own ownership. A view returned from a local owner is an
error. Rebinding a parameter creates a local owner when necessary; it does not
consume the caller's binding.

The compiler plan records initialization, moves, loans, and cleanup on the
executable control-flow graph. An owner cannot be released, replaced, or leave
scope while a view remains live. Cleanup must not make an invalid program appear
valid by erasing a loan. Rust MIR checks source lifetimes before elaborating
cleanup, then checks the result. Native lowering obtains a fresh plan while
borrowing that exact body and its type contracts immutably, so transforms cannot keep using an earlier
proof after changing the body. Ownership failures retain the offending source
span and surface as `E000303` diagnostics.

Constructor storage starts with uninitialized fields, not with an owned value
in every slot. That state follows internal construction bindings and control-flow
joins. Initial assignment does not release an old field, retention visits only
initialized fields, and incomplete construction releases only its initialized
contents. Reading or returning an uninitialized field is rejected with a span.
This prevents undefined placeholder pointers from reaching storage callbacks.

Storage glue is authorized by registered definition identity in the universal
type contracts. A function name resembling compiler glue grants no authority.
The backend implements the registered operation; it does not decide which
values own fields or invent additional destructors.

A copied enum header shares one payload owner. Its active fields are initialized
and destroyed once, when that payload's final owner releases it. Retaining or
releasing a header adjusts payload ownership, not each field's ownership. This
matters when a borrowed callable mutates a payload field: every header observes
the updated field, without separately claiming to own that field. Storage invokes
the compiler-provided active-payload destructor on final release.

Source buffer ownership uses the ownership library’s shared MLIR pipeline.
The C storage registry remains a bootstrap compatibility implementation, not the
source of truth for source buffer lifetimes. See the [migration audit](../../../sev_compiler/frontend/ownership/AUDIT.md).

The hosted provider is `core.memory/native/memory.h`; its external ABI and MLIR
generic allocator adapters are in `memory.c`. Native runtime and system helpers
route allocation through this provider. `core.storage/native/storage.c` owns the
initialized-storage registry, reference accounting, and destructor callbacks.
The bootstrap's old `owned.c` and `owned.h` are compatibility includes of that
implementation. Raw allocations and foreign resources are not silently adopted
as owned values. Foreign release callbacks remain responsible for foreign
allocations.

The source compiler also rechecks executable CFG availability and stack-view
escapes at its lowering entry. Its existing semantic move/loan checking supplies
destructor calls; typed buffer aliases still flow through MLIR's buffer ownership
analysis. This is an explicit remaining integration boundary: its plan is not yet
an implementation of all Rust MIR place/loan checks, and neither checker yet
models every foreign callback lifetime or arbitrary partially initialized
collection. Those cases must not be described as universally verified.

Validation lives in the MIR ownership tests, native method/enum regressions, and
`tests/sev_compiler/storage_reclamation.py`. The latter checks live bytes,
recursive callbacks, shared payloads, and registry growth under sanitizers.
