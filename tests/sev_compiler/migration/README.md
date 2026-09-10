# Complete source-language migration acceptance

Current implementation results and outstanding work are recorded in
[RESULTS.md](RESULTS.md) and [INVENTORY.md](INVENTORY.md).

Run from the repository root after building the source compiler:

```sh
(cd sev_compiler && ../package.pkg/debug/sev build)
python3 tests/sev_compiler/migration.py
python3 tests/sev_compiler/migration.py Gate3SourceSyntax
```

Alternatively build by running `sev build` in `sev_compiler`. Set
`SEVERIAN_SOURCE_COMPILER` to select a different source compiler executable.
The runner never invokes the Rust seed on a test subject. It checks that the
source compiler executable's SHA-256 stays unchanged throughout each test.
LLVM/MLIR tool overrides match the other source-compiler acceptance runners.

These are **30 ordinary tests, five per migration gate**. Unsupported features
fail; none are skipped or marked as expected failures. Several tests contain
multiple assertions or related negative cases to establish a single contract.
The tests are intended to be committed before their implementation.

| Gate | Five acceptance cases |
| --- | --- |
| 1. Baseline and boundary | Scalar behavior; callable/record behavior; imported aliases/members; deterministic output; inspection of the active CFG pipeline |
| 2. Definitions and capabilities | Inherited G contracts; counterfeit G rejection; direct/indirect CFG escapes; contextual compiler words; absent versus explicit values and required fields |
| 3. Source syntax | Imported novel operator; longest symbols and word boundaries; source precedence/associativity; contract renaming and respelling; conflicting imports |
| 4. Semantic execution | Edit semantic helper without rebuilding; generic operator implementations; assignment evaluation once; writable-place requirements; lazy operands and user truth |
| 5. Canonical CFG | Nested loops/match/returns; user grammar constructing a branch; resolved provenance and edge consistency; double-terminator rejection; typed error propagation |
| 6. Library and retirement | Real int/float sources; collection/string protocols; numeric conversions; removing a source contract disables an operation; known legacy dispatch/topology removed |

The static retirement check is supplemental evidence about known old paths.
Renaming a fallback to evade it does not satisfy the migration. Behavioral
tests establish that source definitions are actually used. These thirty tests
are a required acceptance floor, not an exhaustive proof of the language or a
replacement for the existing callable, diagnostic, library, and bootstrap suites.
Ownership cleanup, malformed CFGs beyond the included case, wider numeric
formats, and the rest of the migration inventory still require relevant tests
as those paths are completed.

## Source contract exercised

Compiler contracts are ordinary `trait Name: G` declarations. `symbol: Y`
accepts a source symbol, not a string naming an enum variant. Named compiler
terms such as `Value`, `Place`, `Pure`, `LeftToRight`, and `Branch` resolve in
the compiler context. `operator <symbol>[G:Name]` uses the ordinary
generic-parameter/constraint syntax: `G` is the parameter and `Name` resolves
to a grammar contract. The parameter spelling does not grant a capability.

An abstract `semantic` requirement obtains its concrete body from the type's
operator implementation. When the G contract supplies `semantic`, an operator
signature can bind that implementation without duplicating its body. Those
semantic methods can call ordinary typed helpers. The helper-mutation test
requires executing that source behavior during compilation/instantiation.

The assignment test observes a side-effecting index and RHS. Their traces must
occur once, in the declared order, and the original collection must be updated.
The control-flow fixture deliberately uses remainder `%` for its parity check;
the expected totals are authored independently of compiler output.

## Agent IR acceptance boundary

`sev_compiler test SUBJECT --emit agent-ir --sysroot ROOT` returns JSON from
the same resolved CFG used for MLIR emission. This is a new inspection boundary
required by the migration, not a dump synthesized independently from source.
Fields beyond those below are allowed.

```json
{
  "schema_version": 1,
  "stage": "cfg",
  "definitions": [
    {
      "id": "opaque-definition-identity",
      "name": "qualified.source.name",
      "kind": "grammar",
      "bases": [],
      "metadata": {"overflow": null, "short_circuit": false},
      "source": {"path": "relative/or/absolute.sev", "start": 0, "end": 100}
    }
  ],
  "functions": [
    {
      "definition": "function-definition-identity",
      "entry": "opaque-block-identity",
      "blocks": [
        {
          "id": "opaque-block-identity",
          "parameters": [],
          "operations": [],
          "terminator": {
            "kind": "Return",
            "grammar": "opaque-definition-identity",
            "source": {"path": "subject.sev", "start": 0, "end": 10},
            "successors": []
          }
        }
      ]
    }
  ]
}
```

Definition IDs are unique, nonempty strings; their encoding is not prescribed.
`bases` contains resolved definition IDs. A function body's `definition`
references a definition with `kind: "function"`. Grammar definitions have
`kind: "grammar"`. Declaration `source` spans cover the whole declaration,
using half-open UTF-8 byte offsets. Source absence serializes as JSON `null`;
explicit `false` stays `false`.

Every CFG terminator has a resolved grammar origin and a source origin,
including implicit terminators. Each successor is an object with `target` and
`arguments`; argument counts agree with the destination's `parameters`.
Canonical block operations expose their `kind`; nested legacy `If`, `Loop`,
and `Choose` bodies are not a second executable topology.

The definition-removal test copies the source sysroot, removes the actual
declaration identified by the emitted IR, and recompiles with the unchanged
binary. A source-resolution diagnostic is required. It must not use a hidden
compiled-in Add contract or scalar operator fallback.
