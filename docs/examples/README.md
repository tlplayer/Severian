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

