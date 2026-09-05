# Scalar, function, and UTF-8 bootstrap compiler

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
Unsigned `u8` values and comparisons are supported. `char` literals represent
Unicode scalars; `string` literals represent immutable UTF-8 byte views.
Functions have concrete typed parameters, positional calls, local bindings,
and scalar, string, or unit returns. Forward calls, recursive calls, and `if`/`else`
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

Unsupported declarations and expressions produce diagnostics. Classes,
general collections, floating-point values, parameter defaults,
named arguments, global captures, reassignment, and loops remain outside this
slice. This executable does not yet compile itself. The broader driver and
generic semantic pipeline remain separate unfinished work.

## Source-compiled strings and output

The compiler reads and compiles these `.sev` sources along with each input:

- [`universal/primitive/string/core.sev`](../universal/primitive/string/core.sev)
  implements byte counts, checked byte access, UTF-8 decoding, character counts,
  character indexing, and equality.
- [`universal/primitive/char/encoding.sev`](../universal/primitive/char/encoding.sev)
  supplies shared Unicode scalar arithmetic. Character literals use the same
  decoding functions; `char/utf8.sev` exposes character-to-codepoint conversion.
- [`library/system/io/src/text.sev`](../../library/system/io/src/text.sev)
  implements `print(string)` by traversing those bytes and appending a newline.
  Its only native IO boundary is the platform C library's `putchar(i32)`.

These are source inputs, not prebuilt IR or C string helpers. Source functions
retain their resolved callable identities through HIR and MIR into MLIR.
`@mlir("dialect.operation")` on a bodyless typed function declares a direct
operation binding; `@c(symbol="name")` declares a scalar C boundary. The emitter
consults that metadata instead of recognizing library function names. Direct
MLIR bindings are a small compiler-library escape hatch: the upstream MLIR
verifier remains responsible for checking dialect-specific signatures.

Literals become constant `memref.global` byte arrays and values use
`memref<?xi8>` views. Returning a literal or passing a view through a function
therefore preserves its storage lifetime. This is the first byte-array/view
representation, not general array or slice syntax. String concatenation,
allocation, owned-string construction, and general collection lowering remain
ahead. Traversal currently uses recursion and is intended for small inputs.

The full [`universal/primitive/string.sev`](../universal/primitive/string.sev)
owning-string class does **not** compile through this slice yet. Its UTF-8
continuation and leading-width helpers now delegate to the extracted core;
the rest still needs class, pointer, allocation, loop, and conversion support.
The existing seed-facing IO package root is also unchanged. `text.sev` is the
source-bootstrap implementation, not a replacement for all IO overloads.

Single-quoted literals contain one Unicode scalar. Double-quoted strings and
characters support `\n`, `\r`, `\t`, `\\`, `\"`, and `\'`. Other escapes,
including NUL and numeric Unicode escapes, are diagnosed. The seed's input
runtime still uses C strings; this slice does not support embedded NUL input.

Relative source imports, optionally `as alias`, load ordinary modules before
analysis. Imports resolve relative to the importing file and cycles are
diagnosed. Imported modules contain function declarations; their tests are not
run. These are development source locators: dependency aliases, manifests,
lockfiles, and `.pkg` consumption from the
[package golden path](../../docs/examples/05-building/README.md) are not yet
implemented by this bootstrap driver. It defaults to the repository working
directory for library sources; use `--sysroot /path/to/Severian` elsewhere.

## Basic examples shortlist

The acceptance runner uses these existing files unchanged:

| Example | Mode | Coverage |
| --- | --- | --- |
| `docs/examples/05-building/src/math.sev` | build | Typed integer function and return |
| `docs/examples/03-testing/01-basics/01-ordinary-and-named.sev` | test | Calls, comparisons, early returns, ordinary and named tests |
| `docs/examples/00-getting-started/01-hello.sev` | build | String literal and source-library output |

```sh
/tmp/sev-bootstrap-driver test --emit mlir \
    docs/examples/03-testing/01-basics/01-ordinary-and-named.sev > /tmp/clamp.mlir
mlir-opt-21 --verify-each /tmp/clamp.mlir -o /tmp/clamp.verified.mlir
```

```sh
/tmp/sev-bootstrap-driver build --emit mlir \
    docs/examples/00-getting-started/01-hello.sev > /tmp/hello.mlir
mlir-opt-21 /tmp/hello.mlir --verify-each --convert-scf-to-cf \
    --convert-arith-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts -o /tmp/hello.llvm.mlir
mlir-translate-21 --mlir-to-llvmir /tmp/hello.llvm.mlir > /tmp/hello.ll
clang-21 /tmp/hello.ll -o /tmp/hello
/tmp/hello
# hello, severian
```

Next are general byte arrays/slices and the owning-string representation,
followed by broader function examples and collection libraries.

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
It also compiles the character encoding and string core modules directly,
checks exact hello/Unicode output, string and character return values, bounds
failures, relative imports, and invocation with an explicit sysroot. The native
outputs link without Severian C string or IO helpers.
Artifacts are retained in `target/acceptance` beneath this package.
