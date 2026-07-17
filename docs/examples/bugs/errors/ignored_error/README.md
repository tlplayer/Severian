# Ignored error

`Result` is not an ignorable value. The caller must bind its successful value
with `?=`, return the exact result, handle it with `switch`, or use an explicit
discard form whose reason is visible in review.

Mitigation: **compile-time**.
