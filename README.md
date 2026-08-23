# Severian

Severian is a compiled systems language with Python-like syntax,
Rust-like safety, Go-style concurrency, and MLIR/XLA-backed execution.

## Install

```sh
git clone https://github.com/tlplayer/Severian.git
cd Severian
cargo install \
    --path compiler/boundaries/driver \
    --force

sev doctor
```
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
- [Contributing](CONTRIBUTING.md)