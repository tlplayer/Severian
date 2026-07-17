# Ignored error

`Result` is not an ignorable value. The caller must propagate it with `?=`,
handle it with `switch`, or use an explicit discard form whose reason is visible
in review.

Mitigation: **compile-time**.
