# Canonical examples

Every `.sev` file below this directory is a standalone documentation example
and an independent validation source root. The examples do not carry package
manifests or compiler fixtures. Imports such as `import tensor` are resolved by
the shared package context in `test/validation/examples`.

Run the complete corpus with:

```text
sev test test/validation/examples
```

That package reaches this directory through a relative `linked` symlink. It
discovers every `.sev` file, runs ordinary and compiler tests, executes examples
with `main`, and writes a report using these canonical paths.

To audit the same corpus with the source-written `sev_compiler`, from the
repository root run:

```sh
target/debug/sev build sev_compiler
python3 tests/sev_compiler/examples.py
```

The audit builds and executes each example, then independently runs files that
declare tests. Unsupported features and test modes are reported as failures.
Builds exclude test bodies, so a successful build alone does not validate an
example's tests. Each invocation retains native artifacts, stdout/stderr logs,
and Markdown/JSON reports under `sev_compiler/target/examples/`. Adjacent output
fixtures are compared exactly when present. Use a file or directory argument
to narrow the audit, or `--timeout` to adjust its per-command timeout.
