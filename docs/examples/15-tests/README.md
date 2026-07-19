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
  surface is transitive through the call graph. A caller inherits the scenarios
  of its dependencies and adds scenarios belonging to its own layer.
- Compatible modes compose with `and`. The API-contract comma rule does not
  apply to test modes.

The files in this directory demonstrate each form and representative use cases.

`when function return/throw value` is test-only syntax. It is valid in both an
ordinary `test:` and a `test with chaos` block, and is a compile-time error
outside a test. That boundary prevents production code from intercepting a
function and replacing its behavior.

```sev
test with chaos "read results":
    chaos.add(when read return None)
    chaos.add(when read return Failure(PermissionDenied))

    for event in chaos:
        result = read()

test with chaos "read exceptions":
    chaos.add(when read throw PermissionDenied)
    chaos.add(when read throw TimedOut)

    for event in chaos:
        result = read()
```

Multiple named tests divide the catalog into understandable slices. Each event
runs independently. Higher tests inherit reachable lower-level scenarios and
add scenarios belonging to their own layer:

```text
wrapper
├── wrapper's own scenarios
└── read
    ├── return None
    ├── return Failure(PermissionDenied)
    ├── throw PermissionDenied
    └── throw TimedOut
```

Handling an event does not remove it from the transitive catalog. Injecting it
at the caller also verifies the dependency's recovery and the caller's behavior
after that recovery.
