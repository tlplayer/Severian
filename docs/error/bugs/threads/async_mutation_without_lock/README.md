# Async mutation without a lock

Arguments passed to async work are frozen by default. A mutable receiver cannot
cross the task boundary unless the call also transfers its lock capability.
This keeps compound state such as an account's `balance` and `status` consistent
when multiple children update it.

Mitigation: **compile-time**. Use `with self and lock` to give the child
exclusive mutable access for the call, or pass only frozen data with `with self`.
The child owns the lock while it runs; the waiting parent does not retain it.
