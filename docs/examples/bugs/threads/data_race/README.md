# Data race

Tasks may run on CPU worker threads or accelerator submission threads; scheduling
does not weaken ownership. Mutable raw values cannot cross a task boundary. A
captured value must be frozen, atomic, or guarded by a mutex.

Mitigation: **compile-time** through task-capture checks. Frozen values permit
shared reads; atomic values permit synchronized scalar mutation; mutexes guard
larger mutable state.
