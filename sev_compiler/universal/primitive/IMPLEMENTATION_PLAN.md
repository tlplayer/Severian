# Primitive implementation plan

Status: implementation in progress. The [capability ledger](CAPABILITIES.md) records measured compiler boundaries, exact baseline diagnostics, naming/provider decisions, and implemented slices of P1/P2/P4/P6/P7. No full milestone is complete; existing bodies and successful seed checks do not establish native API coverage.

The objective is to make every type and public operation in this directory executable through the Severian source compiler, with behavior defined in `.sev`. The companion [collections plan](../collections/IMPLEMENTATION_PLAN.md) builds on these milestones. Full string completion crosses that boundary because `String.split` returns `list[string]`.

## Current boundary

The active [source pipeline](../../boundaries/driver/src/pipeline/source.sev) loads grammar contracts, text I/O, `numeric/conversion.sev`, `numeric/operators.sev`, and `primitive/collections.sev`. It does not load the complete primitive directory as its authoritative prelude.

The active [semantic analyzer](../../frontend/semantic/src/callable.sev) still contains scalar dispatch and explicit rejection paths for generic record layouts and owned string fields in records. [Ownership analysis](../../transforms/mir/src/ownership.sev) records CFG availability and aliases; general loans, element initialization, and destruction need further implementation. Consequently, enabling these files requires coordinated compiler work, not just replacing missing method bodies.

Keep the seed able to build the source compiler throughout migration. Ordinary operations, conversions, collection behavior, and formatting belong in Severian. The compiler boundary should describe target layout, verified primitive operations, allocation/FFI, and generic language mechanisms.

## Complete scope

Documentation examples and test fixtures are not additional public types. The coverage ledger must include every public constructor, operator, method, generated specialization, and compile-failure test for these declarations.

| File | Types and contracts to finish | Milestone |
|---|---|---|
| `primitive.sev` | `PrimitiveCategory`, `PrimitiveRepresentation`, `Primitive`, `PrimitiveAlias[T]`; `IndexError`, `UnicodeError`, `TypeConversionError`, `ValueError` | P1 |
| `bool.sev` | `bool`, `Boolean`, `Truth`, `View`, `Copy`, `Borrow`, `Default`, `Hash`, `Print`, `ConstParam`, `Structural`, `BitwiseAssign` | P1–P2 |
| `int.sev` | `IntegerArithmetic`, `IntegerShift`, `IntegerBitwiseAssign`, `Integer`, `SignedInteger`, `UnsignedInteger`; `signed_int`, `unsigned_int`, `int` families and their shared behavior | P2 |
| `int.sev` | `i8`, `i16`, `i32`, `i64`, `i128`, `isize`, `u8`, `u16`, `u32`, `u64`, `u128`, `usize` | P2 |
| `float.sev` | `FloatArithmetic`, `Floating`; `f8` and `float` families and shared behavior; `f8e4m3fn`, `f8e5m2`, `bf16`, `f16`, `f32`, `f64`, `f128` | P3 |
| `char.sev` | `char_storage`, `Character`, `char[u8]`, default `char[u32]` | P4 |
| `char/encoding.sev`, `char/utf8.sev` | Scalar validation, UTF-8 widths/decoding, literal codepoint bridge | P4 |
| `pointer.sev` | `AddressSpace` (`generic`, `host`, `device`, `global`, `shared`, `local`), `pointer[T]`, `*[T]` alias, unsafe operations | P5 |
| `array.sev` | `FixedContainer[T,N]`, `Array[T,N]`, `array[T,N]` | P6 |
| `slice.sev` | `Slice[T]`, `slice[T]`, array slicing extension | P6 |
| `string.sev` | `String`, `string`, UTF-8 construction, operators, parsing and formatting conversions | P7, then C3/P8 |
| `string/core.sev`, `string/format.sev` | Existing byte storage and scalar formatting support; migration adapters | P7–P8 |
| `numeric/conversion.sev`, `numeric/operators.sev` | Generated numeric bridges, `NativeInteger`, `NativeSignedInteger`; reconcile with canonical families | P2–P3, P9 |
| `collections.sev` | Temporary `list[T] = buffer[T]`, generated list operations, `Range` and `range` | P6, C1, P9 |

## P0 — Establish the capability and dependency ledger

1. Record separate results for seed parsing/checking, source-compiler checking, native execution, and Agent IR for each file. Preserve exact diagnostics and baseline failures.
2. Enumerate the public API from parsed declarations, including generated members. Do not mistake prose examples or source tests for successful implementation.
3. Resolve declaration inconsistencies before broad prelude activation: duplicate `Truth` declarations in `bool.sev`; `class float: Float` versus the declared `Floating`; shared family behavior versus same-named union identity; `Print.Print` versus concrete `print`; layout vocabulary (`byte`, `data_size`, `Size`); `bytes[N]`; nongeneric `Range` versus `Range[usize]`.
4. Resolve imports for supporting `Type`, constructor/conversion metadata, `Error`, `Clone`, iteration, allocation, and formatting. Referenced names such as `decimal.format_float` and `Sed` need actual providers. A missing provider is a tracked dependency, not an implicit builtin.
5. Record decisions on arithmetic overflow, division by zero, invalid shifts, allocation failure, ordering, and conversion policy. Preserve behavior already specified by source tests; add explicit contracts where it is unspecified.

Exit: a per-file result ledger, resolved public naming, and one minimal fixture per missing compiler capability. This is the first implementation change to make.

## P1 — Source-defined types, contracts, and conversions

Work areas: `frontend/parser`, `frontend/modules`, `frontend/semantic`, `universal/type`, `universal/declaration`, and the MIR/MLIR transforms.

- Register primitive metadata by declaration identity. Distinguish representation from behavior and family constraints from runtime tagged unions. An `int` family specialization must not acquire a runtime wrapper.
- Support type/constant parameters, defaults, `Self`, trait composition, properties, transparent aliases, generic class layouts, extensions, generated family members, and compile-time metadata expressions used by these files. Cache specialization by definition and canonical arguments.
- Resolve overloaded constructors/operators and the `<=>` conversion graph through normal declarations. Implement identity, lossless `->`, approximate `~>`, policy selection, error results, ambiguity checks, and `:conversions` metadata derived from those same edges.
- Keep truth dispatch separate from conversion. Validate paired `if`/`else` operators and evaluate receivers and indexed places once, including compound assignment.
- Wire error classes through typed propagation and cleanup. Unify capability definitions so imports do not introduce competing `Copy`, `Hash`, or `Default` identities.
- Validate MLIR operation signatures, predicates, attributes, layouts, and target availability at the boundary. New source types using an existing representation must not require a name-based compiler case.

Exit: a user-defined generic type and alias can use traits, properties, an operator, and `<=>` under the source compiler. Editing its conversion body changes native output with the compiler binary unchanged. Invalid constraints and ambiguous conversions produce source-span diagnostics.

## P2 — Boolean and all integer widths

- Activate `bool` behavior, Boolean operators, truth, default/copy/hash/printing, and its declared conversion relations. Preserve `i1` computation versus one-byte addressable storage.
- Instantiate the integer family for all twelve concrete members. Derive pointer-sized widths and alignment from target layout, including cross-target compilation.
- Implement and validate every arithmetic, comparison, bitwise, shift, assignment, sign, absolute-value, rotation, endian, and conversion operation declared in `int.sev`.
- Handle exceptional arithmetic before emitting operations with undefined or poison results. Test signed minimum divided by `-1`, signed minimum absolute value, zero divisors, oversized shifts, narrowing, and signed/unsigned boundaries under the selected policy.
- Implement lossless/approximate conversion selection from source metadata; preserve unsupported-conversion rejection. Bring up numeric-to-numeric conversions first. Endian results depend on P6 storage; textual conversions depend on P7.
- Retain a narrowly scoped adapter for existing compiler consumers until canonical declarations provide their exact behavior.

Exit: native edge-case tests for every width; a generated conversion matrix across signedness and widths; Boolean truth-versus-conversion tests; correct target layout in emitted IR. P2 completes only after its P6/P7-dependent methods also pass.

## P3 — Every floating format

Bring up `f32`/`f64` first, then `f16`/`bf16`, then both `f8` formats and `f128`. The later formats remain required scope.

- Implement format metadata and the complete arithmetic, floor-division/modulo, comparison, classification, rounding, elementary/transcendental math, fused multiply-add, and conversion API.
- Preserve IEEE unordered comparison behavior, signed zero, subnormals, and the declared truth behavior of NaN. Respect the finite-only range of `f8e4m3fn`.
- Audit actual target/toolchain support per operation. Use an explicit runtime or software implementation where native lowering is unavailable. Specify widening and rounding behavior; never silently substitute `f32`/`f64` storage or precision.
- Validate all float-to-float and integer/float conversion edges, including precision/range loss and nonfinite values. Printing/parsing is completed with P7.
- Resolve the difference between partial floating order and the total ordering required by ordered containers. Do not automatically make NaN-bearing floats valid B-tree keys.

Exit: boundary and rounding tests for all seven formats, conversion matrices, and native or explicit software execution on the supported host. Unsupported targets receive precise capability diagnostics and remain visible in the support matrix. MLIR emission alone is insufficient.

## P4 — Characters and UTF-8

- Implement transparent `char[u8]` and `char[u32]` layouts and all `Character` methods, comparisons, hashing, and conversions. `char[u8]` is a byte/code unit; default `char[u32]` is a Unicode scalar.
- Share scalar validation and encoding arithmetic with literal decoding and string construction. Validate the complete UTF-8 sequence, including continuation bytes, overlong encodings, surrogates, truncation, and values beyond U+10FFFF.
- Define whether character classification methods are ASCII-only or Unicode-aware, and provide the corresponding source tables when needed.

Exit: all scalar boundaries and UTF-8 widths pass; invalid input produces `UnicodeError`; literals, explicit construction, iteration, and conversion agree.

## P5 — Memory and ownership foundation

- Implement `pointer[T]` metadata, aliases, equality, address access, construction, load/store, and element-scaled offset. Resolve the existing intrinsic declarations through one typed memory boundary.
- Define address-space lowering per target, legal casts, alignment, pointer width, null behavior, and unsafe requirements. Distinguish logical address spaces even where a host target maps several to the same representation.
- Implement allocation/free, checked allocation-size arithmetic, layout queries, and failure handling. Pointer-to-integer round trips must not silently create a safe borrow or extend a lifetime.
- Extend MIR with initialized places, moves, shared/exclusive loans, owner/region tracking, per-element destruction, and cleanup on ordinary return, early return, errors, and partial construction.
- Define `Copy` consistently for resource owners. `Array` currently claims `Copy` while owning an allocation: implicit duplication must not duplicate ownership of a raw pointer. Keep explicit deep `copy()`/`clone()` distinct unless the language specifies otherwise.

Exit: native allocation/layout probes and destructor counters pass; double free, use after move, escaping borrows, and invalid safe pointer access are rejected. Backend diagnostics cover unsupported address spaces.

## P6 — Arrays, slices, ranges, and iteration

- Implement `array[T,N]` with the allocation-backed representation its source specifies. Finish fixed-size construction, default initialization, checked access, size/alignment, filling, explicit copying, iteration, and destruction. Validate constant lengths, including zero and allocation-size overflow.
- Implement `slice[T]` as a borrowed owner-relative region. Normalize nested subslices. Infer read/write loan requirements from use, permit proven disjoint regions, and conservatively reject unresolved overlap. A slice never frees its owner and cannot outlive it.
- Validate bounds before pointer arithmetic. Reconcile signed array indexing and unsigned slice indexing; decide empty `first`/`last` behavior and align signatures/tests.
- Replace the temporary `Range` with a coherent generic range contract: exclusive stop, signed steps where applicable, zero-step rejection, overflow-safe termination, and iteration protocol integration.
- Reconcile `Iterator`/`iterator`, `yield`, and `library/core/collections/traits/src/iterator.sev`. Distinguish yielded owned values from references into collection storage; do not return a borrowed reference to a temporary tuple.
- First enable current `Copy + Default` element cases. After initialized-slot/drop support, permit owned elements where operations allow them; attach `Default` and copying constraints only to operations that actually require them.

Exit: native fixed-storage tests plus negative lifetime/overlap tests; exact allocation/drop counts; nested slices preserve owner identity; iterator lifetime and early-exit cleanup tests. This unlocks collections C1–C3.

## P7 — Core owned string and all scalar text conversions

- Move from the existing byte-view helpers to the declared owned UTF-8 representation, preserving a temporary adapter for active compiler text consumers.
- Implement every existing concrete constructor, byte/character accessor, range operation, concatenation/repetition, comparison, search predicate, iteration, hash, clone, and drop. Keep byte length and scalar length distinct; define indexing in Unicode scalars, not grapheme clusters.
- Make `raw()` a correctly tracked read-only borrow in safe code: writes must not invalidate string UTF-8 or cached lengths. Defend against size overflow, aliasing, partial allocation failure, and double destruction.
- Implement integer parsing/formatting for every width and supported base, including signed minima and malformed/out-of-range input.
- Supply the missing decimal formatter and a bounded, correctly rounded float parser. The current repeated multiply/add parser needs rounding, exponent-overflow, extreme-exponent, and performance coverage; simple small-value round trips do not establish correctness.
- Connect scalar `string(value)` and string-to-scalar construction through `<=>`, including error propagation. Preserve the Token conversion behavior requested earlier: `string(token)` returns spellings such as `=`, `.`, and `+=` without a `token_text` helper.

Exit: native Unicode/storage tests and scalar parse/format tests across every integer and float type, including signed zero, nonfinite values, subnormals, and halfway rounding cases. The compiler and diagnostics run through the new string provider before its adapter is retired.

## P8 — Finish the declared String contract after collections C3

`String` declares `upper`, `lower`, `split`, `join`, `find`, `match`, and `sed` without corresponding concrete methods. These remain implementation work even when P7 passes.

- Specify and implement Unicode casing, splitting/joining, search result indexing, and missing-result behavior. Fix unclear receiver/argument signatures in the trait.
- Enable `list[string]` using owned-element collection support; do not force `string` into trivial `Copy` semantics to satisfy the temporary list constraint.
- Provide an explicit regex/match contract and the referenced `Sed` type in an appropriate supporting module. Specify matching, captures, replacement, errors, and resource limits before implementing them. Pin Unicode data/version where tables are used.

Exit: every `String` requirement has a checked implementation and native tests. Regex/Sed and collection-returning methods cannot be silently excluded from primitive completion.

## P9 — Prelude migration and acceptance

1. Replace temporary numeric and collection imports in the source pipeline with canonical providers in dependency order. Prevent duplicate names and import cycles.
2. Remove superseded scalar conversion/operator fallbacks and generated buffer-list adapters only after their consumers and regression fixtures use the new definitions. Track each remaining seed dependency with an explicit removal condition.
3. Build the source compiler natively using the checkout seed, then run primitive and consumer tests through that source compiler. Run `sev build --emit agent-ir` from `sev_compiler` with the checkout tool on PATH to inspect call targets, representations, and cleanup. Agent IR emission does not replace the native build.
4. Reuse the unchanged-binary fixture pattern in `tests/sev_compiler/migration.py`: edit or remove a source operation/conversion in a temporary sysroot and prove the behavior changes or fails with the compiler binary unchanged.
5. Run relevant lexer/parser/source-contract/diagnostic regressions and the compiler build. Record existing unrelated failures separately; do not turn historical failures into untracked exclusions.

Completion requires the full coverage ledger to pass through source checking and native execution on supported targets, all primitive operations to resolve to source definitions or documented target boundaries, and all temporary adapters to be removed or explicitly retained solely for seed bootstrap.
