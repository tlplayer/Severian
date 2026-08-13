# Memory safety and development diagnostics

Severian treats static memory safety and runtime memory diagnostics as separate,
complementary layers.

Safe Severian code is checked to prevent use-after-free, double free, invalid
aliasing, and unsynchronized shared mutation. Those guarantees do not imply
that every program has bounded memory use or releases every allocation before
process exit. Intentional process-lifetime retention, reference cycles, unsafe
native code, and foreign libraries remain distinct concerns.

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
