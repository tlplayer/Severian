# Severian compiler

This is the first executable source-to-MLIR compiler slice. Its implementation
is Severian: the Rust seed builds the compiler executable, and that executable
then reads and compiles new source files without invoking Rust or `sev`.

```sh
cd sev_compiler
sev build
sev_compiler ../docs/examples/00-getting-started/01-hello.sev
sev_compiler build --emit mlir ../docs/examples/00-getting-started/01-hello.sev -o /tmp/hello.mlir
sev_compiler test ../docs/examples/03-testing/01-basics/01-ordinary-and-named.sev
```

The package target is `target/host/dev/bin/sev_compiler`. The development command
in `../bin/sev_compiler` runs that artifact and selects this checkout's source
libraries. Linking it into a directory on PATH makes subsequent `sev build`
results immediately available as `sev_compiler`. The Rust seed remains `sev`.
Direct executable invocation accepts `--sysroot /path/to/Severian`, or reads
`SEVERIAN_SYSROOT`; its fallback `..` supports invocation from this package.
Bare file invocation runs the program. `build` writes a native executable;
`--emit mlir` writes MLIR; `check` performs source analysis and MLIR construction.
The hosted native boundary uses LLVM/MLIR 21 tools and the existing environment
variables for selecting them. Compiler code lives under the normal frontend,
transform, and boundary modules; `bootstrap/src` only retains compatibility imports.


The pipeline uses the shared source, lexer, parser, and universal expression
and operation models. `frontend/semantic/src/callable.sev` registers callable signatures and lowers
complete bodies. `transforms/mir/src/callable.sev` lowers calls and storage
operations. `transforms/mlir/src/emit/callable.sev` uses the typed
`MlirProgram` builder. Structural validation precedes terminal text printing.

The target symbol/operator/grammar separation and exclusive grammar capability
for CFG construction are specified in [GRAMMAR.md](GRAMMAR.md). The executable
path still uses structured MIR; source grammar execution and CFG capability
enforcement remain a separate migration.

Supported input consists of integer/boolean bindings, signed `i8`, `i16`,
`i32`, `i64` types (`int` defaults to `i64`), parentheses, unary `+`/`-`/`not`,
binary `+`/`-`/`*`, integer comparisons, Boolean equality, and `assert(condition)`.
Unsigned `u8` values and comparisons are supported. `char` literals represent
Unicode scalars; `string` literals represent immutable UTF-8 byte views.
`f64` (`float`) supports literals, arithmetic, comparisons, and signed zero.
Decimal/exponent spellings survive into typed MLIR attributes without passing
through the seed's lossy decimal formatter. `type(value)` produces the concrete
type's display name while preserving evaluation of the value expression.
Functions have concrete typed parameters, positional and keyword calls, local bindings,
and scalar, string, or unit returns. Forward calls, recursive calls, and `if`/`else`
with early returns are supported. A source `main` takes no parameters and
returns unit or an `i32` exit status.

Conditional expressions (`left if condition else right`) evaluate only the
selected branch and require matching value types. `:=` declares a mutable
binding; `=`, `+=`, `-=`, and `*=` can update it without changing its type.
Immutable literal globals can be read inside functions. Other global captures
remain unsupported. Module initializers execute in source order before `main`.

`test --emit mlir` generates an executable test entry for ordinary and named
tests. It does not invoke a source `main` implicitly. Test modes such as
integration, property, or benchmark are not implemented by this runner.
`build --emit mlir` excludes test bodies.

Scalar compiler tests can use `test with compiler` with `accept:` and `reject:`
statement blocks. Each case gets isolated bindings and is checked without
executing its body. Other test modes and diagnostic-name matching remain ahead.

Behavioral tests live beside the source they exercise: ordinary `test` blocks
in `string/core.sev`, `string/format.sev`, `char/encoding.sev`, `char/utf8.sev`,
and `library/system/io/src/text.sev`. Run those files directly with
`test --emit mlir`. The test subject has its own lookup scope, so even a
prelude implementation can be tested without duplicate declarations.
Compiler-language regression inputs live next to scalar semantic analysis in
`frontend/semantic/src/scalar/tests/`. The Python acceptance runner supplies
the external MLIR/native toolchain and stdout/diagnostic checks; test bodies
do not need separate files to keep them out of builds.

The [source lexer](frontend/lexer/README.md) represents matched source as `Lx` (Lexeme)
values and classified parser input as `To` (Token) values, before `Y` symbols and `G`
descriptor registration. Its built-in rules are
compiled from Severian; loading arbitrary imported lexical-rule bodies at compiler
runtime remains unfinished.

The executable loads syntax from `universal/grammar/contracts.sev` and discovers
imported `trait Name: G` declarations before parsing dependent bodies. Source
symbols, precedence, associativity and inherited syntax can extend the language
without rebuilding the executable. Explicit `[G:Name]` bindings resolve through
the shared definition environment. A bodyless operator can bind a contract's
typed semantic method, including calls to ordinary source helpers; its signature
must agree with the implementation. Source contract regressions run with
`python3 tests/sev_compiler/source_contracts.py` and the migration runner's
`Gate3SourceSyntax` class.

`universal/operator/scalar.sev` still supplies fallback lowering descriptions
for unmigrated operations. Compile-time execution of compiler-semantic methods,
capability enforcement and the canonical CFG pipeline remain unfinished; typed
callable binding does not complete those migrations. Calls retain resolved
`DefId` and `FunctionId` identities through HIR and MIR.

Literal parameter defaults and concrete overloads are supported. Keyword
arguments retain source evaluation order before being arranged for the callee.
Variadic `*values: V` functions specialize for the supplied argument types;
`for value in values` expands over that pack. Parameters after the pack are
keyword-only. This initial form requires the pack to be first and does not
support call-site unpacking or arbitrary collection iteration.

Concrete record classes and their instance methods use the same callable body
lowering as free functions. Calls preserve receiver storage, source argument
order, mutations, recursion, local rebinding, and returns. `while` and `range`
loops support `break`, `continue`, and early returns. Imported literal constants
remain visible in their defining callable scopes. Memory views remain in SSA
across branches and loops so buffer ownership follows their lifetimes.

Value `match` statements support literal patterns with an existing equality
implementation and a final `_` wildcard. Every arm starts with `case`,
including `case _:`.
Semantic analysis evaluates the subject once
and lowers arms to typed bindings and `if` branches, preserving source order,
arm-local scopes, mutation, and loop control. Without a wildcard, a match may
fall through; value-returning functions still require a return on every path.
Nonliteral patterns and arms after a wildcard produce deliberate diagnostics.

Unsupported declarations and expressions produce diagnostics. General
collections, generic classes, fallible/enum values, nonliteral parameter
defaults, nonnumeric conversions, and nonliteral/mutable global captures remain
outside this slice. Record string fields require aggregate ownership lowering
and are rejected; recursive value records require indirection. This executable does not yet compile itself. The broader driver remains unfinished; its semantic entry now delegates to the
same declaration registration and callable analysis used by this executable.

CLI lexer, parser, and semantic diagnostics retain their source file identity
through imports and print their code, filename, line, column, source line, and
caret when a span is available. Run `python3 tests/sev_compiler/diagnostics.py`
from the repository root to check this reporting boundary.

## Enumerated numeric conversions

[`universal/primitive/numeric/conversion.sev`](universal/primitive/numeric/conversion.sev)
uses declaration macros in the `-> name[T: binding]()` family-enumeration style
of `int.sev` and `float.sev`. Explicit family invocations generate the supported
integer/integer, integer/float, float/integer, and float/float pairs, including
a native test for each pair. These are expanded by the source compiler; there
is no checked-in table of concrete overload bodies or build-time Python generator.

This bootstrap macro subset emits functions and ordinary tests, takes no
runtime arguments, substitutes scalar type properties, and selects static
`if` branches before semantic analysis. The current registry supplies `i8`,
`i16`, `i32`, `i64`/`int`, `u8`, and `f64`/`float`. It does not yet implement
the full enclosing `extend`/`operator <=>` grammar or the other primitive widths.

Constructor policy classification follows the Rust seed's universal conversion
model: `identity`, `promote`, `checked`, then `lossy`, in increasing order of
permitted change. Integer narrowing is checked by default; explicit `lossy`
integer casts wrap. Integer/float conversions require the lossy policy, selected
automatically when no mode is supplied. Mixed numeric arithmetic promotes to a
common supported type, using `f64` when one operand is floating point.

Source guards reject nonfinite and out-of-range float-to-integer inputs before
the MLIR cast, including in lossy mode. Fractions truncate toward zero; accepted
input ranges are `[minimum, maximum + 1)`. The upper bound is exclusive to avoid
rounding `INT64_MAX` into the invalid value `2^63`. Guard failures currently
terminate through `assert`; recoverable `TypeConversionError` lowering remains ahead.

## Source-compiled strings and output

The compiler reads and compiles these `.sev` sources along with each input:

- [`universal/primitive/string/core.sev`](universal/primitive/string/core.sev)
  implements byte counts, checked byte access, UTF-8 decoding, character counts,
  character indexing, equality, and concatenation. String `+`, `==`, and `!=`
  resolve to those ordinary source functions through the shared operator table.
- [`universal/primitive/char/encoding.sev`](universal/primitive/char/encoding.sev)
  supplies shared Unicode scalar arithmetic. Character literals use the same
  decoding functions; `char/utf8.sev` exposes character-to-codepoint conversion.
- [`library/system/io/src/text.sev`](../library/system/io/src/text.sev)
  implements variadic `print` in source. Byte output uses the platform C
  library's `putchar(i32)`; explicit flushing uses `fflush(NULL)`.
- [`universal/primitive/string/format.sev`](universal/primitive/string/format.sev)
  provides overloaded string conversions shared by `print`, `string(value)`,
  and interpolation. Integer, Boolean, and Unicode character formatting is
  source code. Hosted float conversion uses glibc `strfromd`/`strtod` in the C
  numeric locale, with source-managed buffers and a search for the first
  significant-digit precision that round-trips (up to 17 digits). This is
  `%g` display style, not an exact copy of Python's float display conventions.

```sev
print("count", 42, true, 'λ', 0.5)
print("a", "b", sep="|", end="!", flush=true)
print()
print(f"count={42}; ratio={0.5}; type={type(42)}")
```

`sep` and `end` accept strings or `None`, defaulting to a space and newline.
An empty call writes only `end`. `file=None` selects stdout; custom stream
objects are not implemented. `flush=True` flushes hosted output streams through
`fflush(NULL)`. Values convert using Severian conventions (`true`/`false`, for
example). Interpolation supports expressions and doubled braces; format specs
such as `:.2f`, conversion flags, collection formatting, and custom format
protocols remain unsupported and produce diagnostics.

These are source inputs, not prebuilt IR or C string helpers. Source functions
retain their resolved callable identities through HIR and MIR into MLIR.
`@mlir("dialect.operation")` on a bodyless typed function declares a direct
operation binding; `@c(symbol="name")` declares a scalar C boundary. The emitter
consults that metadata instead of recognizing library function names. Direct
MLIR bindings are a small compiler-library escape hatch: the upstream MLIR
verifier remains responsible for checking dialect-specific signatures.
Bindings for operations with operand groups can provide
`operand_segments="1,0"`; these sizes are checked against the signature and
emitted as a typed `operandSegmentSizes` attribute. This supplies the dynamic
size and symbol groups for `memref.alloc` without an allocator-name special case.

Literals become constant `memref.global` byte arrays and values use
`memref<?xi8>` views. Returning a literal or passing a view through a function
therefore preserves its storage lifetime. This is the first byte-array/view
representation, not general array or slice syntax. Concatenation allocates a
fresh buffer in `.sev`, copies the input bytes, and returns its view. Inputs
remain unchanged, including snapshots retained across reassignment.

Native lowering must run `buffer-deallocation-pipeline` with
`private-function-dynamic-ownership`, followed by
`convert-bufferization-to-memref`, before lowering SCF and memrefs to LLVM.
This tracks ownership through calls and conditional returns and inserts frees;
the raw high-level MLIR does not yet contain them. The acceptance runner uses
this pipeline. General collection lowering and the full owning-string class
remain ahead. Traversal currently uses recursion and is intended for small inputs.

The full [`universal/primitive/string.sev`](universal/primitive/string.sev)
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
[package golden path](../docs/examples/05-building/README.md) are not yet
implemented by this bootstrap driver. It defaults to the repository working
directory for library sources; use `--sysroot /path/to/Severian` elsewhere.

## Examples in prerequisite order

The acceptance runner uses these existing files unchanged:

| Example | Mode | Coverage |
| --- | --- | --- |
| `docs/examples/00-getting-started/01-hello.sev` | build | String literal and source-library output |
| `docs/examples/02-functions/01-basic/01-basic-functions.sev` | build | Calls, arithmetic, branches; prints `large` |
| `docs/examples/02-functions/02-control-flow/07-conditional-expression.sev` | test | Conditional string returns and string equality |
| `docs/examples/00-getting-started/02-variables.sev` | build | Literal global, mutable binding, concatenation; prints `Hello, World!` |
| `docs/examples/00-getting-started/03-printing.sev` | build | Ordered module calls, reassignment, string interpolation |
| `docs/examples/01-types/01-basic/01-primitives.sev` | build | Integer, float, Boolean, and character output |
| `docs/examples/01-types/01-basic/00-constants.sev` | build | Type-first constant declarations and decimal precision |
| `docs/examples/01-types/01-basic/02-inference.sev` | build | Mixed interpolation and type names |
| `docs/examples/02-functions/01-basic/02-signatures.sev` | build | Float defaults, keyword calls, and string conversion |
| `docs/examples/01-types/01-basic/03-conversion.sev` | build/test | Mixed arithmetic, explicit conversions, policies, and a compiler rejection case |
| `docs/examples/05-building/src/math.sev` | build | Typed integer function and return |
| `docs/examples/03-testing/01-basics/01-ordinary-and-named.sev` | test | Calls, comparisons, early returns, ordinary and named tests |
| `docs/examples/03-testing/02-with-tests/08-compile.sev` | test | Accepted/rejected fragments and isolated case bindings |

Every row above emits verified MLIR and runs natively. Build rows have their
exact output checked by the acceptance runner; their integration-test blocks
are not executed by the bootstrap test runner.

Trait requirements use explicit source operators:

```sev
trait Addable:
    operator +(other: Self) -> Self

def twice[T: Addable](value: T) -> T:
    return value + value
```

A method named `add` does not implicitly declare `+`. Trait methods, properties,
and inherited contracts are checked against registered members and operator
signatures. Concrete classes and `extend` declarations supply operator bodies;
`Self` resolves to the receiver type. Type aliases resolve before field/function
signatures, and closed `union` families can share behavior through a class or
extension declaration. Duplicate implementations, cyclic aliases/inheritance,
missing members, incompatible results, and unsatisfied constraints are errors.

The executable and `analyze_with_package_functions()` share
[`definitions.sev`](frontend/semantic/src/definitions.sev) and callable body
analysis. The package entry takes `SemanticDefinitions` as its environment,
replacing the disconnected `TypeContext` prototype. Generic bodies are checked
with symbolic parameters before calls are specialized. `Module.generic_functions`
retains that HIR, including references to trait operator definitions; concrete
instances retain their template identity and ordered type bindings; resolved calls
carry those bindings as a `Substitution`. Their
operator calls refer to concrete implementation definitions before MIR lowering.

[`numeric/operators.sev`](universal/primitive/numeric/operators.sev) supplies
source arithmetic implementations for the currently supported integer storage
classes and `float`. Its `mlir(arith.addi, self, right)` / `mlir(arith.addf, self,
right)` bodies become typed MLIR bindings through ordinary callable lowering.
Constraint checking does not recognize numeric types or conventional method
names. The existing scalar table remains the fallback for operations not yet
migrated to source implementations.

This is not yet the whole primitive library: the complete `int.sev` and
`float.sev` still need conversion graphs, metadata evaluation, additional storage
types, and richer MLIR attributes. Generic class layouts, generic trait/operator
arguments, and predicate constraints remain unsupported. The parser retains
operator generic parameters and bodies so these constructs cannot be mistaken
for an empty trait contract.

```sh
sev_compiler test --emit mlir \
    docs/examples/03-testing/01-basics/01-ordinary-and-named.sev > /tmp/clamp.mlir
mlir-opt-21 --verify-each /tmp/clamp.mlir -o /tmp/clamp.verified.mlir
```

```sh
sev_compiler build --emit mlir \
    docs/examples/00-getting-started/01-hello.sev > /tmp/hello.mlir
mlir-opt-21 /tmp/hello.mlir --verify-each \
    --buffer-deallocation-pipeline=private-function-dynamic-ownership \
    --convert-bufferization-to-memref --convert-scf-to-cf \
    --convert-arith-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts -o /tmp/hello.llvm.mlir
mlir-translate-21 --mlir-to-llvmir /tmp/hello.llvm.mlir > /tmp/hello.ll
clang-21 /tmp/hello.ll -o /tmp/hello
/tmp/hello
# hello, severian
```

The seed fixes exercised by this slice include indexed compound assignments,
imported fallible-result metadata, declaration-scope enum defaults, and the
uncaught-error runtime ABI. Records with a string `message` preserve that
message at the terminal error boundary; enum errors currently report their
type name. Mirror-group reclamation is unchanged and remains bootstrap debt.

Run the acceptance check with LLVM/MLIR 21 tools available:

```sh
python3 tests/sev_compiler/bootstrap_mlir.py
# Parser-to-HIR structure and callable/native regressions:
target/debug/sev test tests/sev_compiler/semantic_ir
python3 tests/sev_compiler/callable_bodies.py
# Also check allocated string lifetimes (run outside a ptrace-based sandbox):
SEVERIAN_SANITIZE=1 python3 tests/sev_compiler/bootstrap_mlir.py
```

It builds the source compiler, feeds it several source files, verifies its
MLIR, lowers that MLIR to native executables, and checks both successful and
failing assertions. It also checks native exit status, forward and recursive
calls, branch-local values, and rejection of invalid signatures, calls, and
return paths. Invalid input must produce diagnostics rather than crash.
It also runs the native tests in the character, string, and IO source modules,
and verifies that build output contains no test functions. It
checks exact hello/Unicode output, string and character return values, bounds
failures, conditional evaluation, mutation, concatenation snapshots, relative
imports, and invocation with an explicit sysroot. Printing checks cover empty
and mixed variadic calls, defaults, keyword evaluation order, pack scopes,
Unicode, integer limits, subnormal/nonfinite floats, and full-precision decimal
literals. The optional sanitizer checks exercise returned/borrowed buffers,
recursive concatenation, and float formatting through the hosted boundary. The native
outputs link without Severian C string or IO helpers.
Artifacts are retained in `target/acceptance` beneath this package.

## Generic IR migration in progress

The shared structural type/interner/inference port lives in
`universal/type/system.sev`, following Rust's `universal/src/type_system.rs`.
Substitution bindings now carry `GenericParamId` and `TypeId` explicitly.
The structural interner is not connected to the compiler entry yet; executable
generics currently use the shared scalar/record identities and declaration environment. The seed now
lowers instance method bodies through ordinary callable/HIR bodies, with
receiver storage preserved across calls. Seed regressions live in
[`method_bodies.rs`](../rust_compiler/boundaries/driver/tests/method_bodies.rs).
The native compiler now uses the same callable flow for methods and functions.
Its regressions live in `frontend/semantic/src/callable/tests/`; run
`python3 tests/sev_compiler/callable_bodies.py` from the repository root.
Set `SEVERIAN_SANITIZE=1` to check the emitted binaries with sanitizers.
The adjacent structural type tests have a previously reported ownership failure
on a moved value. Bootstrap acceptance now preserves typed diagnostics from
rejected variadic specializations, including the numeric macro regression that
previously escaped as an untyped error. The callable nested-control-flow fixture
uses `%` for parity and independently calculated totals, including 133 for its
early return. The migration inventory and current acceptance boundary are in
[`tests/sev_compiler/migration`](../tests/sev_compiler/migration/README.md).
The scalar/macro path remains present during migration; generic functions now
use source trait requirements, while the broader compiler-term generic system
is still incomplete. Numeric-only macro enumeration is
not the final family/constraint design.
