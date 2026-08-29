# Testing

API ID: `testing.declaration`

Tests, compiler accept/reject cases, properties, benchmarks, chaos, integration, profiling, differential cases, timeouts, skips, repeats, and mocks are first-class syntax and retain source spans.

```sev
test "testing API probe":
    assert(2 + 2 == 4)
```

The test runner consumes typed test declarations rather than parsing names. Current weakness: behavioral symmetry oracles do not yet cover every implemented API group.
