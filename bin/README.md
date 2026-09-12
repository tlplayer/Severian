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

`sev` runs the source compiler in `sev_compiler`. `sev_rust` runs the Rust
seed and remains available when the source compiler needs recovery. Both
launchers resolve their checkout through symlinks and pass through arguments.

Both native compilers implement `--profile` directly:

```sh
sev input.sev --profile
sev_rust build input.sev --profile cpu --build-profile release
sev test input.sev --profile memory
sev run input.sev --profile time
```

Bare `--profile` prints CPU, memory, and time totals. Use `--build-profile release`
to select build settings. See [compiler profiling](../tests/sev_compiler/PROFILING.md)
for stack capture, native accounting, candidate binaries, and report contents.

Install or rebuild the current checkout with:

```sh
python3 tools/compiler/update.py --local
```

This builds an optimized Rust seed and the source compiler, runs a native smoke test, and installs launchers in
`${CARGO_HOME:-$HOME/.cargo}/bin`. Existing commands are backed up before being
replaced. Use `--install-dir DIR` to choose another directory or `--no-install`
to build and verify without changing commands on PATH.

`sev update` and `sev_rust update` fetch the current branch's upstream, fast
forward the checkout, rebuild both compilers, and verify the source compiler.
Local changes and divergent histories stop the update without being discarded.
`update --local` builds current sources without fetching. The Rust command calls
the same updater and can recover a broken source compiler. At the launcher,
`update` is reserved for compiler updates; package dependency updates remain
available through `sev_compiler update`.
