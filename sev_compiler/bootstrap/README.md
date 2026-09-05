# Scalar and function bootstrap compiler

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

Supported input consists of integer/boolean bindings, signed `i8`, `i16`,
`i32`, `i64` types (`int` defaults to `i64`), parentheses, unary `+`/`-`/`not`,
binary `+`/`-`/`*`, integer comparisons, Boolean equality, and `assert(condition)`.
Functions have concrete typed parameters, positional calls, local bindings,
and scalar or unit returns. Forward calls, recursive calls, and `if`/`else`
with early returns are supported. A source `main` takes no parameters and
returns unit or an `i32` exit status.

`test --emit mlir` generates an executable test entry for ordinary and named
tests. It does not invoke a source `main` implicitly. Test modes such as
integration, property, or benchmark are not implemented by this runner.
`build --emit mlir` excludes test bodies.

Operator signatures and lowering descriptions live in the shared source table
`universal/operator/scalar.sev`. Semantic analysis and MLIR lowering consult
that table; the emitter does not maintain a second operator-symbol switch.
This is an initial homogeneous-scalar lowering form, not the complete
compiler-term or extensible lowering protocol. Calls retain resolved `DefId`
and `FunctionId` identities through HIR and MIR.

Unsupported declarations and expressions produce diagnostics. Imports,
classes, collections, floating-point values, strings, parameter defaults,
named arguments, global captures, reassignment, and loops remain outside this
slice. This executable does not yet compile itself. The broader driver and
generic semantic pipeline remain separate unfinished work.

## Basic examples shortlist

The acceptance runner uses these existing files unchanged:

| Example | Mode | Coverage |
| --- | --- | --- |
| `docs/examples/05-building/src/math.sev` | build | Typed integer function and return |
| `docs/examples/03-testing/01-basics/01-ordinary-and-named.sev` | test | Calls, comparisons, early returns, ordinary and named tests |

```sh
/tmp/sev-bootstrap-driver test --emit mlir \
    docs/examples/03-testing/01-basics/01-ordinary-and-named.sev > /tmp/clamp.mlir
mlir-opt-21 --verify-each /tmp/clamp.mlir -o /tmp/clamp.verified.mlir
```

The next shortlist item is `docs/examples/00-getting-started/01-hello.sev`,
which needs string values and output. After that, the full basic-functions
example combines those features with the calls and conditionals implemented
here. Collections follow those prerequisites.

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
failing assertions. It also checks native exit status, forward and recursive
calls, branch-local values, and rejection of invalid signatures, calls, and
return paths. Invalid input must produce diagnostics rather than crash.
Artifacts are retained in `target/acceptance` beneath this package.
