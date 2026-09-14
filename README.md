# Severian

Severian is a compiled systems language with Python-like syntax,
Rust-like safety, Go-style concurrency, and MLIR/XLA-backed execution.

## Install

```sh
sev --version
sev init hello
cd hello
sev run
```

Contributors building Severian itself should use Cargo as described in
[CONTRIBUTING.md](CONTRIBUTING.md). To build this checkout and replace an older
Cargo-installed `sev` command with it, run:

```sh
sev --help
sev --version
sev update #Updates the compiler
```

The source installer uses `${CARGO_HOME:-$HOME/.cargo}` by default, so the new
binary replaces an older `$HOME/.cargo/bin/sev` instead of being hidden behind
it on `PATH`. Set `SEV_CARGO_INSTALL_ROOT` to select another Cargo installation
root. Building the compiler and installing a release remain separate workflows.

For a checkout using the `bin/sev` launcher, `sev update` fetches and builds the
latest upstream compiler with the release profile. A cold smoke test must finish
within 90 seconds before the update is accepted. Use `sev update --local` to build uncommitted local
work without fetching. The update prints the source compiler's version, UTC
build date, source checkout path, and full Git commit hash. `sev --version`
reports the same details later; local modifications are marked explicitly.
These details are embedded at link time, including when building the source
compiler directly. Source archives without Git metadata report an unknown
commit. Failed builds or smoke tests preserve the previous compiler.

Local builds write generated files to `package.pkg/`. Cargo places the seed
binary at `package.pkg/debug/sev`; Severian packages place binaries at
`package.pkg/<platform>/<profile>/bin/`. Reusable compilation outputs and input
fingerprints live in `package.pkg/build/units/`. Repeating the same compilation
unit restores its output before loading the prelude;
changing a run/test output path does not force compilation. `sev clean` removes
package artifacts. This cache does **not** provide compiled dependency or prelude
reuse across different programs: library builds still produce source bundles.

Local source-compiler publications live at
`~/.severian/packages/registry/<name>/<version>/`. `SEVERIAN_HOME` overrides
`~/.severian`; `SEVERIAN_REGISTRY` overrides the registry directory directly.
Explicit dependencies take precedence, locked dependencies retain their selected
versions, and unpinned standalone imports prefer the latest published version
over a sysroot source package. Registry discovery changes invalidate standalone
compilation caches.

Run `bash test/validation/packages/compiled_reuse.sh` to check the full compiled
package contract. It currently rejects missing semantic interfaces and source
recompilation. Build/publication is bounded to 90 seconds per invocation, fresh
consumers to 5 seconds, tests to 15 seconds, and the entire check to 300 seconds.
The command keeps evidence in a printed temporary directory and exits nonzero
on failure. These are acceptance bounds, not measured compiler performance.

## Bootstrap compiler checkpoint

```sh
sev build \
    sev_compiler/bootstrap \
    --bin sev-bootstrap-driver
```

The source-written bootstrap compiler currently emits MLIR for a scalar
subset. See [the bootstrap instructions](sev_compiler/bootstrap/README.md)
for source-to-MLIR execution and verification. Full self-hosting remains in
progress.

## Try Severian

```sh
sev docs/examples/00-getting-started/01-hello.sev
```

Or create a project:
```sh
sev new hello
cd hello
sev run
```
## Examples

Start with [`docs/examples`](docs/examples/README.md).

Examples are the executable language reference. They cover syntax,
packages, concurrency, systems programming, tensors, MLIR/XLA,
and larger integration examples.

## Documentation

- [Examples](docs/examples/README.md)
- [Language reference](docs/LANGUAGE.md)
- [Packages](docs/PACKAGES.md)
- [Tooling](docs/TOOLING.md)
- [Compiler architecture](docs/COMPILER_ARCHITECTURE.md)
- [Agent IR](docs/READMEs/AGENT_IR.md)
- [Contributing](CONTRIBUTING.md)
