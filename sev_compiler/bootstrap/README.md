# Scalar bootstrap compiler

This is the first executable source-to-MLIR compiler slice. Its implementation
is Severian: the Rust seed builds the compiler executable, and that executable
then reads and compiles new source files without invoking Rust or `sev`.

```sh
cargo build -p severian-driver --bin sev
target/debug/sev build sev_compiler/bootstrap --bin sev-bootstrap-driver -o /tmp/sev-bootstrap-driver
/tmp/sev-bootstrap-driver build --emit mlir sev_compiler/boundaries/driver/tests/fixtures/int_add.sev > /tmp/int_add.mlir
mlir-opt-21 --verify-each /tmp/int_add.mlir -o /tmp/int_add.verified.mlir
```

The pipeline uses the shared source, lexer, parser, and universal expression
and operation models. `frontend/semantic/src/scalar.sev` resolves bindings and
checks scalar types. `transforms/mir/src/scalar.sev` creates SSA operations.
`transforms/mlir/src/emit/scalar.sev` lowers those operations through the typed
`MlirProgram` builder. Structural validation precedes terminal text printing.

Supported input consists of module-level integer/boolean bindings, signed
`i8`, `i16`, `i32`, `i64` types (`int` defaults to `i64`), parentheses, unary
`+`/`-`, binary `+`/`-`/`*`, `==`/`!=`, and `assert(condition)`. Unsupported
declarations and expressions produce diagnostics. This executable does not
yet compile functions, imports, classes, collections, or itself. The broader
driver and generic semantic pipeline remain separate unfinished work.

The seed fixes exercised by this slice include indexed compound assignments,
imported fallible-result metadata, declaration-scope enum defaults, and the
uncaught-error runtime ABI. Records with a string `message` preserve that
message at the terminal error boundary; enum errors currently report their
type name. Mirror-group reclamation is unchanged and remains bootstrap debt.

Run the acceptance check with LLVM/MLIR 21 tools available:

```sh
python3 tests/sev_compiler/bootstrap_mlir.py
```

It builds the source compiler, feeds it several source files, verifies its
MLIR, lowers that MLIR to native executables, and checks both successful and
failing assertions. Invalid input must produce diagnostics rather than crash.
Artifacts are retained in `target/acceptance` beneath this package.
