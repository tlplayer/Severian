# Concurrency examples

These examples cover structured task ownership, awaiting results, task-returned
results, runtime-owned work, shared-state locking, bounded channels, and channel
selection. Every `.sev` file has a matching native-output fixture.

The assertion-backed cases are:

| File | Behavior under test |
| --- | --- |
| `01-async-await.sev` | Basic task results and multiple self-owned children. |
| `09-nested-task-tree.sev` | Nested structured task trees and independent sibling trees. |
| `10-locked-shared-mutation.sev` | Serialized shared mutation and lock release after awaiting. |
| `11-buffered-fan-in.sev` | Multiple producers, typed buffered channels, and receive ordering. |

The remaining examples exercise result propagation, runtime ownership,
implicit structured completion, channel send/receive, and multi-channel
switching through their executable `main` functions.
