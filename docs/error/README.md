# Compiler error catalog

The format, namespace, applicability rules, and backend translation boundary are
defined in [the diagnostics contract](../DIAGNOSTICS.md).

Each diagnostic has a stable code and a Severian source file under its primary
category. Filenames use `EXXXXXX-description.sev` for errors and
`WXXX-description.sev` for warnings. The source itself is the documentation:
`#` comments explain why the compiler reports it, where detection happens, and
the smallest normal repair.

These files intentionally demonstrate rejected or warned-about programs. They
are documentation fixtures, not runnable examples.

The `runtime` category contains programs that compile successfully and then
produce a structured `E0009xx` diagnostic during native execution. Severian uses
one six-digit `E######` namespace across compilation and execution so engineers, agents,
editors, and `sev explain` do not need separate code conventions.

Runtime diagnostics use the source-focused `user` view by default. Set
`diagnostics = "internal"` in `package.toml`, or pass
`sev run --diagnostics=internal`, to include the native artifact, protocol
version, and canonical source path. Native failures that have not yet been
classified are reported as E000990 instead of leaking a bare signal or core-dump
message. E000990 automatically includes a symbolic stack trace in the default
view, so developers do not rerun the command with a diagnostics flag merely to
obtain crash evidence. Each migrated runtime failure replaces that fallback
with its stable, specific E0009xx code.

`sev errors` prints the complete registered E-code index. `sev explain EXXXXXX`
opens the longer causes-and-fixes entry for one code.

The driver test suite discovers every catalog file and requires its filename's
diagnostic to be the first reported code. Each result must include a source
file, line, column, snippet, and a registered `sev explain` entry. Diagnostics
that require package context are tested in that context: E000103 is loaded as an
invalid dependency and E000201 is compiled with `type-safe = true`.

`sev build` checks independent package sources before emitting artifacts. It
reports up to 50 errors by default so a package or an automated repair agent can
address a useful batch at once without receiving cascades from later compiler
stages. Configure the bound and structured output with:

```sh
sev build --max-errors 20
sev build --message-format json
```

A direct `sev file.sev` invocation stays focused. Executable sources run;
function-only sources compile to a linkable LLVM module; accelerator kernels
compile to the artifact selected by their `@compile` policy.

Package boundaries may progressively require concrete types with:

```toml
[package]
type-safe = true
```

That mode rejects declarations which would silently default to `Any`. Explicit
`Any` remains available when a dynamic boundary is intentional.
