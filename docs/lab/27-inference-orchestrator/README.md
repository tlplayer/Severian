# Native inference node

This package is a finite, executable single-node inference service. It batches
six requests through a capacity-limited channel, runs two native workers, uses a
symbolic model graph backed by ranked tensor kernels, retries one injected worker
failure, reports health and queue pressure, and crosses a real TCP loopback
socket as its transport probe.

The example intentionally terminates so both its service output and its attached
tests can be compiled and executed in CI. The worker pool itself uses the same
pthread tasks, condition-variable channels, ownership checks, and model runtime
as a long-running server.

Run the native service directly:

```sh
sev docs/examples/27-inference-orchestrator/main.sev
```

Or build its Cargo-style package:

```sh
cd docs/examples/27-inference-orchestrator
sev build
./target/debug/inference-node-example
```

`sev build` walks the dependency graph first and validates each library before
writing its reusable `target/debug/deps/lib<package>.sevi` artifact. The final
application compilation consumes those artifacts. Package-local library tests
stay in their owning package instead of being linked into the application test
binary.

Compile and execute the node's three native tests independently:

```sh
sev compile-tests main.sev -o /tmp/inference-node-tests
/tmp/inference-node-tests
```
