# Tests

`test` is Severian's single test declaration. `with` selects specialized runner
behavior before the optional test name:

```sev
test:
test "ordinary named test":
test with property "generated values and shrinking":
test with bench "warmup and measurement":
test with chaos "fault injection":
test with property and chaos "generated inputs under injected failures":
```

- Ordinary tests provide deterministic examples and regression checks.
- Property tests generate typed inputs and shrink failures.
- Benchmark tests perform warmup and repeated measurements while retaining
  correctness assertions.
- Chaos tests inject failures at every reachable chaos point. Their failure
  surface is transitive through the call graph, so a caller does not duplicate
  the failure inventory of its callees.
- Compatible modes compose with `and`. The API-contract comma rule does not
  apply to test modes.

The files in this directory demonstrate each form and representative use cases.
