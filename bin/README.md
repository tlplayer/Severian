# Compiled examples

Run `sev test test/validation/examples` to validate the canonical examples.
The validation package compiles each `.sev` file with an isolated module
identity and executes files that declare `main`. Verified executable artifacts
remain invocation-local under the validation package's `target` directory.

```text
docs/examples/00-getting-started/01-hello.sev
bin/examples/00-getting-started/01-hello
```

Adjacent `.stdout` and `.stderr` files are exact output fixtures when present.
Unsupported lowering, crashes, nonzero exits, and output differences fail the
validation run.
