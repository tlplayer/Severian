# Severian

Severian is a compiled systems language with Python-like syntax,
Rust-like safety, Go-style concurrency, and MLIR/XLA-backed execution.

## Install

```sh
curl -LsSf https://severian.dev/install.sh | sh
sev --version
sev init hello
cd hello
sev run
```

Pin an exact release with:

```sh
curl -LsSf https://severian.dev/install.sh | SEV_VERSION=0.1.0 sh
```

The installer downloads a prebuilt archive from the canonical
[GitHub Releases](https://github.com/tlplayer/Severian/releases) page, verifies
its SHA-256 checksum, and installs it under `~/.local`. It never invokes Cargo
or builds Severian from source. Set `SEV_ATTESTATION=required` to additionally
require GitHub Sigstore provenance verification through `gh`.

Currently tested release targets are `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`. Optional accelerator stacks are not required for
installation; inspect them separately with `sev doctor`.

Contributors building Severian itself should use Cargo as described in
[CONTRIBUTING.md](CONTRIBUTING.md). Building the compiler and using the compiler
are intentionally separate workflows.

## Nightly Bootstrapped compiler

```sh
sev build \
    sev_compiler/boundaries/driver \
    --bin sev-bootstrap-driver
```

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
