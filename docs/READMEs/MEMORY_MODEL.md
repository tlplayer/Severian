# Memory safety and development diagnostics

Severian treats static memory safety and runtime memory diagnostics as separate,
complementary layers.

Safe Severian code is checked to prevent use-after-free, double free, invalid
aliasing, and unsynchronized shared mutation. Those guarantees do not imply
that every program has bounded memory use or releases every allocation before
process exit. Intentional process-lifetime retention, reference cycles, unsafe
native code, and foreign libraries remain distinct concerns.

## Function arguments and ownership

Passing an argument does not implicitly clone it. Scalar values use their native
value representation; collections, objects, strings, and tensors pass their
existing runtime handle. The ownership pass infers each parameter's strongest
effect from the function body:

- read-only use is a shared `view`;
- mutation or `borrow` is an exclusive borrow;
- `move` is an ownership transfer;
- `clone` is the explicit copy operation.

The inferred effect is enforced at every call site. Loans end after their last
use rather than at the end of the lexical scope. Consequently, mutation after a
view's last use is accepted, while using a value after a call to an inferred
consuming parameter is rejected. Today these checks establish aliasing and
use-after-move safety; they do not insert lifetime-driven heap releases.

Effect inference currently examines direct uses in each function body. It does
not yet compute a transitive call-graph fixed point or serialize effects in
package interfaces. A call whose implementation is unavailable to the current
program, including an `@c` declaration, defaults to a shared view unless the
argument explicitly requests `borrow` or `move`.

The native runtime's collection `clone` copies the collection header and item
pointer array. It is a shallow container copy, so referenced/boxed elements are
shared. Other value kinds currently lower `clone` to the same runtime value and
need type-specific clone implementations before they can promise a deep copy.

See `docs/examples/05-ownership-borrowing/04-inferred-parameter-effects.sev`
for an executable profile example.

## Development commands

```sh
sev test --profile
sev test --memory
sev test --profile --memory
sev test --profile --memory --leaks
```

Profile tests report:

- `time_ns`: elapsed monotonic time for the test body.
- `allocated_bytes`: bytes allocated during the test body.
- `allocations`: allocation operations during the test body.

The allocation measurements are cumulative deltas, not current live heap or
resident-set size. They are stable inputs for profile contracts and regression
checks, but a falling value does not prove that memory was reclaimed.

`--memory` compiles and runs the selected tests with AddressSanitizer and
UndefinedBehaviorSanitizer by default. `--sanitizer thread` and
`--sanitizer memory` select their respective standalone runtimes. These dynamic
checks cover compiler/runtime defects and unsafe native boundaries that static
ownership checks cannot observe.

`--leaks` enables LeakSanitizer and therefore requires the address sanitizer.
It is explicit because the current native value runtime intentionally retains
some allocations for the life of the process, and because LeakSanitizer is not
available in every restricted development environment. A reported leak is a
bug-finding signal to classify, not evidence that safe Severian code permitted
memory corruption.

The next level of leak precision is an owner-aware native allocation ledger:
live bytes, peak live bytes, allocation sites, ownership identity, and explicit
intentional-retention annotations. Until that lands, leak findings are
sanitizer-backed and profile `memory` continues to mean total allocated bytes.
