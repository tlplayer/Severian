# Tests

`test` is Severian's single test declaration. `with` selects specialized runner
behavior before the optional test name:

```sev
test:
test "ordinary named test":
test with property "generated values and shrinking":
test with bench "warmup and measurement":
test with chaos "fault injection":
test with profile "runtime and allocation bounds" -> TestResult with
{
    defer 1ms < time < 2ms -> exception("runtime outside range", location, vars),
    defer memory < 32mb -> exception("memory limit exceeded", location, vars),
    defer allocations < 1000 -> exception("allocation limit exceeded", location, vars),
}:
    measured_operation()
test with property and chaos "generated inputs under injected failures":
```

- Ordinary tests provide deterministic examples and regression checks.
- Property tests generate typed inputs and shrink failures.
- Benchmark tests perform warmup and repeated measurements while retaining
  correctness assertions.
- Chaos tests inject failures at every reachable chaos point. Their failure
  surface is transitive through the call graph. A caller inherits the scenarios
  of its dependencies and adds scenarios belonging to its own layer.
- Profile tests expose `time` in nanoseconds, `memory` in allocated bytes, and
  `allocations` as a count. Unit literals such as `1ms` and `32mb` are normalized
  by the compiler. Bounds are checked after the measured body, including lower
  bounds, so stubbed work cannot satisfy `1ms < time`. The runner always prints
  all three measurements, even when the test does not declare a profile
  contract.
- Compatible modes compose with `and`. The API-contract comma rule does not
  apply to test modes.

The files in this directory demonstrate each form and representative use cases.

Compiler tests currently accept only `accept:` and `reject:` cases. Their
fragments have already passed lexing and parsing, so they test semantic
compilation rather than lexer/parser diagnostics. A named diagnostic binding
such as `reject error:` is reserved for the future diagnostic object model and
is rejected for now instead of being silently ignored.

Run every test with `sev test`. In a directory without `package.toml`, this
discovers `.sev` files in the current directory and its subdirectories. Use
`sev test path/to/directory` to select another directory, or `sev test file.sev`
to select one file. The native source compiler tests each discovered file in a
separate process, continues after failures, and prints a file summary. It skips
`package.pkg`, `errors`, hidden entries, and symlinks. An empty batch or any
failed file returns a nonzero exit status. A directory with `package.toml`
uses its existing package test configuration.

Add `--profile` for a CPU/memory/time breakdown
of the invocation, including compilation and test execution. Use
`sev test --profile cpu` for CPU stacks or `sev test --profile memory` for
allocation stacks. These options measure the invocation and do not filter
which tests run or enable sanitizers.

`when function return/throw value` is test-only syntax. It is valid in both an
ordinary `test:` and a `test with chaos` block, and is a compile-time error
outside a test. That boundary prevents production code from intercepting a
function and replacing its behavior.

```sev
test with chaos "read results":
    chaos.add(when read return None)
    chaos.add(when read return Failure(PermissionDenied))

    for event in chaos:
        result ?= read()

test with chaos "read exceptions":
    chaos.add(when read throw PermissionDenied)
    chaos.add(when read throw TimedOut)

    for event in chaos:
        result ?= read()
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
