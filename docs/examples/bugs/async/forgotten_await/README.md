# Forgotten await

Starting work creates a `Task[T]`; discarding it silently loses completion and
failure information. Detached background work must use an explicit `detach`
operation that states its error sink and cancellation policy.

Mitigation: **compile-time**.
