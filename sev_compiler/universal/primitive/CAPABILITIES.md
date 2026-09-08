# Primitive capability ledger

This ledger accompanies [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). No
milestone is marked complete by a successful seed check or by an existing body.
P0 still needs full generated-member coverage and the unresolved contracts below;
P1–P9 remain open. Initial observations were made on 2026-09-08.
The [measured checkpoint](MEASUREMENTS.md) lists every primitive file and capability
fixture with separate stage results.

## Reproduce the measurements

From the repository root, after building the source compiler with the checkout
seed:

```sh
cd sev_compiler
PATH="$PWD/../target/debug:$PATH" sev build
PATH="$PWD/../target/debug:$PATH" sev build --emit agent-ir
cd ..
python3 tests/sev_compiler/primitives.py --snapshot tests/sev_compiler/fixtures/primitives/checkpoint.json
python3 tests/sev_compiler/primitive_behaviors.py
```

`primitives.py` measures every `.sev` file in this directory and the minimal
fixtures in `tests/sev_compiler/fixtures/primitives`. It never rebuilds a compiler.
`--fail-on-unsupported` makes any unsuccessful stage fail the command; the default
records the incomplete migration without disguising failures as expected tests.

Artifacts are under `sev_compiler/target/primitive-ledger`: `results.json` records
compiler/source hashes, exact commands and exit codes; each subject directory
contains unmodified stdout/stderr, seed AST, parsed declaration inventory, source
MLIR, Agent IR, and native artifacts where those stages succeed. A timeout,
compiler crash, missing executable, rejected source and blocked native run are
different results. A native pass runs existing tests, which may be empty; it does
not establish coverage of every operation in that file.

`declarations.json` comes from seed AST declaration nodes and byte spans, not
regular expressions over source examples. It retains methods, constructors,
operators, properties, aliases, enum variants and tests; declarations inside
tests are labeled separately. Seed AST is currently a debug serialization, and
the reader reports an inventory error if its expected node/span format changes.
Source Agent IR retains expanded definitions and origin spans when source checking
succeeds. Unexpanded canonical family generators remain an explicit coverage gap.
The top-level `truth_box` helper is explicitly classified as a fixture. The
quadruple-backtick conversion example in `bool.sev` is not a seed-parsed operator
implementation, and the `map` example inside its documentation is not a type.

The exact initial diagnostics, before these changes, are preserved in
[`baseline-before.json`](../../../tests/sev_compiler/fixtures/primitives/baseline-before.json).
That initial snapshot covers seed parsing/checking and source checking only; it
does not claim native results or record an unavailable original compiler digest.
The refreshed [checkpoint](../../../tests/sev_compiler/fixtures/primitives/checkpoint.json)
records all six stages, compiler/source hashes and inline diagnostics. Full AST,
expanded declaration and native artifacts remain in the reproducible output directory.

## Changes with executable acceptance

- P1 prerequisite: single-quoted block strings use the same decoding and
  indentation behavior as double-quoted block strings. Module documentation is
  ignored by import loading; example declarations inside it never enter the API.
- P1 prerequisite: direct numeric/conversion, numeric/operator and collection
  prelude subjects retain their original scope and are loaded once. Relative
  input paths are resolved against the OS working directory, independently of
  the inherited `PWD` environment variable.
- P1 generation: static conditions in generated operator bodies are pruned just
  as they are in generated functions and tests. Unsigned specializations do not
  type-check signed-only negative literals from a discarded branch.
- P1 specialization limits count instantiated generics and packs, not unrelated
  ordinary declarations in an expanding prelude. A 600-function native fixture
  exercises the boundary without removing the specialization limit.
- P2 adapter: `/`, `//` and `%` run through generated `.sev` implementations for
  the existing `i8`, `i16`, `i32`, `i64` and `u8` storage types. Native acceptance
  checks normal results, signed minima, zero divisors and unchanged-binary source
  edits. This does not add the seven remaining integer types.
- P4 helper: shared two-, three- and four-byte decoders validate every supplied
  byte, shortest encoding, surrogate exclusion and U+10FFFF. Literal decoding and
  string decoding consume these same functions. Errors still terminate through
  `assert`; typed `UnicodeError`, generic characters and Unicode classification
  remain open.
- P4 adapter: all six character comparisons resolve to source codepoint
  comparisons for the active default character representation.
- P6 adapter: `Range` detects signed overflow before adding its step and returns
  the exclusive-stop sentinel. It is still the temporary, nongeneric `int` range.
- P7 adapter: string addition and equality now have ordinary source operator
  declarations, enabling compound addition while preserving old byte views.
  The complete owned `string` class remains unimplemented.

## Naming and provider decisions

| Subject | Decision / current provider | Remaining dependency |
| --- | --- | --- |
| `Truth` | One declaration in `bool.sev`; duplicate removed | Shared import identity and paired dispatch validation |
| `Floating` | Shared `float` behavior names the declared `Floating` contract | Full family specialization and every floating format |
| Float representation | `PrimitiveRepresentation.FloatBits(bits, exponent_bits, fraction_bits, supports_infinity)` supplies the variant used by all seven declarations | Validate metadata by declaration identity and target |
| `Print` | Required method is lowercase `print`, matching concrete methods | Capability imports and all implementations |
| Boolean test naming | `clone`/`display` are requested by source tests, but the class currently exposes `copy`/`print` | Supply or reconcile those methods, correct `Eror` in short-circuit tests, and resolve conversion prose against actual parsed declarations |
| Integer endian methods | Canonical names are `little_endian`, `big_endian`, `native_endian`; hash/native-endian callers use those names | Byte storage and endian lowering |
| Family identity | `int`/`float` are closed compile-time families; shared behavior must specialize onto concrete members without a runtime wrapper | Unify the same-named class/family declaration mechanism with canonical generic layouts |
| Layout | `universal/type/type.sev` owns `Size.Fixed`, `Size.Pointer`, `Size.Dynamic` | Reconcile `byte`, `data_size`, `1B` expressions and lowercase `Size.pointer` with the metadata/byte-count distinction |
| `bytes[N]` | Fixed bytes must use the same fixed-storage implementation as `array[u8,N]` | Define the alias and distinguish it from the unrelated `bytes[T]()` layout query |
| `Range[usize]` | Canonical ranges require generic element type and overflow-safe directional steps | Current provider is nongeneric `Range`; do not reinterpret it as unsigned |
| `Type`, `Constructor`, conversion policy | `primitive.sev` imports `type/type.sev`, which imports `type/conversion.sev` | Source enum/fallible/owned-field support, compile-time constructor metadata |
| `Error` | `library/core/error/src/lib.sev` declares the diagnostic contract; `grammar/contracts.sev` currently supplies a separate marker trait | Resolve the two providers to a single runtime error identity |
| `Copy`, `Hash`, `Default`, `Clone` | First three are declared in `bool.sev`; no canonical `Clone` provider is imported here | Extract shared identities and specify copy versus explicit deep clone |
| Iteration | Active protocols are in `collections.sev`; collection trait candidate is `library/core/collections/traits/src/iterator.sev` | Unify `Iterator`/`iterator`, owned/reference yields, cleanup |
| Allocation | Current byte-view allocation uses typed `memref.alloc` bindings | Typed raw-memory provider, failure handling, initialized slots and destruction |
| Decimal formatting | `string/format.sev` uses hosted `strfromd`/`strtod` for the active f64 adapter | The canonical `decimal.format_float` provider is missing; all other formats remain required |
| `Sed`, regex, casing | No canonical provider is supplied by the current imports | Specify regex/capture/replacement/resource limits and pin Unicode tables in P8 |

Missing providers are work items, not implicit builtins. Broad prelude activation
must wait for these rows and the compiler fixtures; adding empty traits would not
resolve the required behavior.

## Behavior contracts

| Operation | Contract | Current validation boundary |
| --- | --- | --- |
| Integer `+`, `-`, `*` | Existing unflagged MLIR integer operations wrap at the representation width | Active widths only; a checked arithmetic policy would be a separate change |
| Integer `/` | Truncate toward zero; reject zero and signed minimum divided by `-1` | Source guards before `arith.divsi`/`arith.divui`; bootstrap assertion failure |
| Integer `//` | Floor toward negative infinity, with the same exceptional operands as `/` | Source quotient/remainder correction avoids a target-specific floor-divide lowering |
| Integer `%` | Truncating remainder, with dividend sign; reject zero; remainder by `-1` is zero even at signed minimum | Source early return avoids undefined `arith.remsi` operands |
| Shift | Require `0 <= count < bits`; reject invalid counts before lowering | Contract selected for canonical P2; active implementation remains missing |
| Signed `abs` | Reject signed minimum when a positive result cannot fit | Contract selected for canonical P2; current canonical `math.absi` body still needs a guard |
| Numeric conversion | Preserve identity/promote/checked/lossy policy selection; checked narrowing rejects range loss; explicit lossy integer conversion wraps | Existing conversion adapter; recoverable `TypeConversionError` and full `<=>` graph remain missing |
| Allocation | Reject size overflow before allocating; failure preserves previous owner contents and destroys initialized elements once | Contract for P5, not implemented by an unchecked allocation binding |
| Ownership | Owning allocations cannot acquire implicit shallow `Copy`; explicit deep copy/clone remains a distinct operation | Array's current `Copy` claim must be removed when canonical ownership is activated |
| Floating order | IEEE comparisons are partial; NaN is unordered and truthy | Full canonical truth/order and a separate total-order key contract remain required |
| Indexing | Check bounds before pointer arithmetic; empty first/last raise `IndexError`; string indices count Unicode scalars | Full typed errors, borrows and owned string representation remain required |
| Range | Exclusive stop; nonzero signed step; stop before next-position overflow | Native tests of the active int adapter, including both limits and minimum step |

## Minimal capability fixtures

The `.sev` fixtures are positive capability requests: a failed stage is retained
as a missing capability, not accepted as implementation. `documentation.sev`
should now pass. `enum`, `generic_record`, `constant_parameter`, `owned_field`,
`conversion_relation`, `constructor_metadata`, `wide_integer`, `float_storage`,
`generic_character` and `fallible_outcome` exercise subsequent gates. Constructor
metadata imports its real provider and also depends on enum support. The existing
[`migration.py`](../../../tests/sev_compiler/migration.py) supplies the broader
trait, ownership, CFG, target and conversion acceptance gates. Per-operation
test-to-definition mappings, all canonical specializations and additional
negative loan/destruction fixtures remain required before P0/P9 can close.
