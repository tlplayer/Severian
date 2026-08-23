# Getting started

These examples establish the basic Severian workflow before introducing the rest of the language.

Run a source file directly:
```sh
sev docs/examples/00-getting-started/01-hello.sev
```
Check a source file without running it:
```sh
sev check docs/examples/00-getting-started/01-hello.sev
```
Run the tests in a source file:
```sh
sev test docs/examples/00-getting-started/01-hello.sev
```
Build a native executable:
```sh
sev build docs/examples/00-getting-started/01-hello.sev
01-hello.exe
```

For a project workflow:
```sh
sev new hello
cd hello
sev check
sev test
sev build
```
sev check parses, resolves, type-checks, and checks ownership. sev test builds and runs Severian tests. sev build runs the project build policy and emits the configured artifact.

To inspect the compiler pipeline without executing the program, pass
`--emit ast`, `--emit hir`, `--emit mir`, `--emit lir`, or `--emit mlir`.
The representation is printed to standard output by default; use `-o PATH` to
write it to a file.
